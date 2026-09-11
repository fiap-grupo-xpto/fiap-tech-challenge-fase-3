import os
import random
import time
from typing import Optional, Tuple

from backend.assistant.schemas import AssistantProviderResult


class BaseAssistantProvider:
    def is_available(self) -> bool:
        raise NotImplementedError()

    def generate(self, system_prompt: str, user_prompt: str) -> AssistantProviderResult:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        return self.generate_prompt(prompt, system_prompt=system_prompt, user_prompt=user_prompt)

    def generate_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> AssistantProviderResult:
        raise NotImplementedError()


class Item1LocalProvider(BaseAssistantProvider):
    def __init__(self, model_dir: str, tokenizer_dir: Optional[str] = None):
        self.model_dir = model_dir
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer_dir = tokenizer_dir
        else:
            self.tokenizer_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        if not self.model_dir or not os.path.isdir(self.model_dir):
            return False
        if not os.path.isdir(self.tokenizer_dir):
            return False
        return True

    def _get_model_and_tokenizer(self):
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import AutoConfig

            base_model_id = os.getenv("ITEM1_BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            tokenizer = None
            tokenizer_errors = []
            for tokenizer_source in (self.tokenizer_dir, self.model_dir, base_model_id):
                if not tokenizer_source:
                    continue
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
                    break
                except Exception as exc:
                    tokenizer_errors.append(str(exc))

            if tokenizer is None:
                raise RuntimeError("Failed to load tokenizer: " + " | ".join(tokenizer_errors))

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            device_map = "auto" if torch.cuda.is_available() else None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = None
            model_errors = []
            for model_source in (self.model_dir,):
                try:
                    _ = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        torch_dtype=dtype,
                        device_map=device_map,
                    )
                    break
                except Exception as exc:
                    model_errors.append(str(exc))

            if model is None:
                try:
                    from peft import PeftModel

                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_id,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                    model = PeftModel.from_pretrained(base_model, self.model_dir)
                except Exception as exc:
                    model_errors.append(str(exc))
                    raise RuntimeError("Failed to load model: " + " | ".join(model_errors))

            self._model = model
            self._tokenizer = tokenizer
            return model, tokenizer
        except Exception as exc:
            self._load_error = str(exc)
            raise

    def generate_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> AssistantProviderResult:
        if not self.is_available():
            return AssistantProviderResult(
                answer_text="Modelo customizado indisponível.",
                backend_used="item1_custom_llm",
                custom_llm_available=False,
                fallback_used=False,
                provider_error="item1 artifacts not available",
            )

        try:
            model, tokenizer = self._get_model_and_tokenizer()

            if system_prompt is not None and user_prompt is not None:
                # Usa o chat template nativo do tokenizer (o MESMO formato usado para
                # montar o texto de treino no notebook de fine-tuning) em vez do texto
                # genérico "System: ...\nHuman: ..." produzido pelo LangChain — sem
                # isso, o modelo era servido em produção com uma distribuição de
                # entrada diferente da que ele viu durante o treino.
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            if hasattr(model, "device") and inputs is not None:
                try:
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                except Exception:
                    pass
            input_length = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Decodifica só os tokens gerados após o prompt (por posição, não por
            # comparação de string) — com chat templates, o prompt formatado contém
            # tokens especiais que `skip_special_tokens=True` remove do texto
            # decodificado, o que quebraria uma checagem de prefixo por string.
            answer_text = tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True
            ).strip()

            return AssistantProviderResult(
                answer_text=answer_text,
                backend_used="item1_custom_llm",
                custom_llm_available=True,
                fallback_used=False,
                provider_error=None,
            )
        except Exception as exc:
            return AssistantProviderResult(
                answer_text="Não foi possível gerar resposta com o modelo customizado.",
                backend_used="item1_custom_llm",
                custom_llm_available=True,
                fallback_used=False,
                provider_error=str(exc),
            )


class GeminiFallbackProvider(BaseAssistantProvider):
    def __init__(self):
        self._client = None
        self._client_error: Optional[str] = None

    def is_available(self) -> bool:
        api_key = os.getenv("GEMINI_API_KEY")
        use_vertex = os.getenv("USE_VERTEX_AI", "false").lower() in ("true", "1", "yes")
        if use_vertex:
            return True
        return bool(api_key)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._client_error is not None:
            raise RuntimeError(self._client_error)

        try:
            from google import genai
        except Exception as exc:
            self._client_error = str(exc)
            raise

        use_vertex = os.getenv("USE_VERTEX_AI", "false").lower() in ("true", "1", "yes")
        try:
            if use_vertex:
                project = os.getenv("VERTEX_PROJECT")
                location = os.getenv("VERTEX_LOCATION", "us-central1")
                self._client = genai.Client(vertexai=True, project=project, location=location)
            else:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY nao configurada e USE_VERTEX_AI nao ativado")
                self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception as exc:
            self._client_error = str(exc)
            raise

    def _generate_with_retry(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "1.5"))

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                client = self._get_client()
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text, None
            except Exception as exc:
                last_error = str(exc)
                message = last_error.lower()
                retryable = any(
                    marker in message
                    for marker in (
                        "429",
                        "quota",
                        "resource_exhausted",
                        "rate limit",
                        "too many requests",
                    )
                )
                if attempt >= max_retries or not retryable:
                    break
                jitter = random.uniform(0.0, 0.25)
                delay = base_delay * (2 ** attempt) + jitter
                time.sleep(delay)

        return None, last_error

    def generate_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> AssistantProviderResult:
        text, error = self._generate_with_retry(prompt)
        if text is not None:
            return AssistantProviderResult(
                answer_text=text,
                backend_used="gemini_fallback",
                custom_llm_available=False,
                fallback_used=True,
                provider_error=None,
            )

        return AssistantProviderResult(
            answer_text="Não foi possível gerar a interpretação pela LLM neste momento.",
            backend_used="gemini_fallback",
            custom_llm_available=False,
            fallback_used=True,
            provider_error=error,
        )


class AutoFallbackProvider(BaseAssistantProvider):
    """Tenta o modelo local (Item 1) primeiro e só recorre ao Gemini quando a geração
    genuinamente falha — não apenas porque a pasta de artefatos existe. `is_available()`
    do Item1LocalProvider só verifica se o diretório existe, não se o modelo carrega; sem
    esta camada, uma falha real de carregamento (dependência ausente, peso corrompido)
    ficava presa no provedor local em vez de cair para o Gemini de verdade.
    """

    def __init__(self, primary: BaseAssistantProvider, fallback: BaseAssistantProvider):
        self._primary = primary
        self._fallback = fallback

    def is_available(self) -> bool:
        return self._primary.is_available() or self._fallback.is_available()

    def generate_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> AssistantProviderResult:
        if self._primary.is_available():
            try:
                primary_result = self._primary.generate_prompt(
                    prompt, system_prompt=system_prompt, user_prompt=user_prompt
                )
            except Exception as exc:
                primary_result = AssistantProviderResult(
                    answer_text="", backend_used="item1_custom_llm", custom_llm_available=False,
                    fallback_used=False, provider_error=str(exc))
            if primary_result.provider_error is None and primary_result.answer_text.strip():
                return primary_result
            if primary_result.provider_error is None:
                primary_result.provider_error = "Empty model response"

            # A tentativa primária falhou mas o fallback pode dar certo — isso não pode
            # apagar o rastro do que deu errado. Sem isto, um "success" via Gemini
            # escondia completamente que o modelo local falhou e por quê, inclusive da
            # auditoria (nada no log indicava que houve uma tentativa anterior).
            try:
                fallback_result = self._fallback.generate_prompt(
                    prompt, system_prompt=system_prompt, user_prompt=user_prompt
                )
            except Exception as exc:
                fallback_result = AssistantProviderResult(
                    answer_text="", backend_used="gemini_fallback", custom_llm_available=False,
                    fallback_used=True, provider_error=str(exc))
            fallback_result.attempted_backend = primary_result.backend_used
            fallback_result.attempted_backend_error = primary_result.provider_error
            return fallback_result

        return self._fallback.generate_prompt(prompt, system_prompt=system_prompt, user_prompt=user_prompt)


class AssistantProviderSelector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_model_dir = os.path.join(base_dir, "llm_finetuning", "artifacts", "llama_medical_lora_model")
        default_tokenizer_dir = os.path.join(base_dir, "llm_finetuning", "artifacts", "tokenizer")
        self.model_dir = os.getenv("ITEM1_MODEL_DIR", default_model_dir)
        self.tokenizer_dir = os.getenv("ITEM1_TOKENIZER_DIR", default_tokenizer_dir)

    def select(self, mode: str) -> BaseAssistantProvider:
        mode_normalized = (mode or "auto").lower()

        if mode_normalized == "gemini_only":
            return GeminiFallbackProvider()

        if mode_normalized == "item1_only":
            tokenizer_dir = self.tokenizer_dir if os.path.isdir(self.tokenizer_dir) else None
            return Item1LocalProvider(self.model_dir, tokenizer_dir)

        return AutoFallbackProvider(
            Item1LocalProvider(self.model_dir, self.tokenizer_dir),
            GeminiFallbackProvider(),
        )
