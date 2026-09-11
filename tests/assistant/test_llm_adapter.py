from backend.assistant.llm_adapter import AutoFallbackProvider, BaseAssistantProvider
from backend.assistant.schemas import AssistantProviderResult


class _FakeProvider(BaseAssistantProvider):
    def __init__(self, available: bool, result: AssistantProviderResult):
        self._available = available
        self._result = result

    def is_available(self) -> bool:
        return self._available

    def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
        return self._result


def _ok_result(backend: str) -> AssistantProviderResult:
    return AssistantProviderResult(
        answer_text="ok",
        backend_used=backend,
        custom_llm_available=True,
        fallback_used=False,
        provider_error=None,
    )


def _failed_result(backend: str, error: str) -> AssistantProviderResult:
    return AssistantProviderResult(
        answer_text="Não foi possível gerar resposta com o modelo customizado.",
        backend_used=backend,
        custom_llm_available=True,
        fallback_used=False,
        provider_error=error,
    )


def test_auto_fallback_uses_primary_when_it_succeeds():
    primary = _FakeProvider(True, _ok_result("item1_custom_llm"))
    fallback = _FakeProvider(True, _ok_result("gemini_fallback"))

    result = AutoFallbackProvider(primary, fallback).generate_prompt("prompt")
    assert result.backend_used == "item1_custom_llm"
    assert result.provider_error is None
    assert result.attempted_backend is None
    assert result.attempted_backend_error is None


def test_auto_fallback_falls_back_when_primary_directory_exists_but_generation_fails():
    """Reproduz o bug relatado: is_available()==True (a pasta existe) não significa que
    o modelo carrega. Antes desta correção, isso ficava preso no provedor local."""
    primary = _FakeProvider(True, _failed_result("item1_custom_llm", "model failed to load"))
    fallback = _FakeProvider(True, _ok_result("gemini_fallback"))

    result = AutoFallbackProvider(primary, fallback).generate_prompt("prompt")
    assert result.backend_used == "gemini_fallback"
    assert result.provider_error is None


def test_auto_fallback_preserves_primary_failure_reason_when_fallback_succeeds():
    """A causa da falha do provedor primário não pode desaparecer quando o fallback dá
    certo — sem isso, um `status: success` via Gemini escondia completamente que o
    modelo local foi tentado e falhou, inclusive da auditoria."""
    primary = _FakeProvider(True, _failed_result("item1_custom_llm", "model failed to load"))
    fallback = _FakeProvider(True, _ok_result("gemini_fallback"))

    result = AutoFallbackProvider(primary, fallback).generate_prompt("prompt")
    assert result.attempted_backend == "item1_custom_llm"
    assert result.attempted_backend_error == "model failed to load"


def test_auto_fallback_surfaces_fallback_error_when_both_fail():
    primary = _FakeProvider(True, _failed_result("item1_custom_llm", "model failed to load"))
    fallback = _FakeProvider(True, _failed_result("gemini_fallback", "no api key configured"))

    result = AutoFallbackProvider(primary, fallback).generate_prompt("prompt")
    assert result.backend_used == "gemini_fallback"
    assert result.provider_error == "no api key configured"
    assert result.attempted_backend == "item1_custom_llm"
    assert result.attempted_backend_error == "model failed to load"


def test_auto_fallback_skips_primary_when_not_available():
    primary = _FakeProvider(False, _ok_result("item1_custom_llm"))
    fallback = _FakeProvider(True, _ok_result("gemini_fallback"))

    result = AutoFallbackProvider(primary, fallback).generate_prompt("prompt")
    assert result.backend_used == "gemini_fallback"


def test_primary_exception_preserved_in_fallback():
    class Broken(BaseAssistantProvider):
        def is_available(self): return True
        def generate_prompt(self, *args, **kwargs): raise RuntimeError("native wrapper failed")
    result = AutoFallbackProvider(Broken(), _FakeProvider(True, _ok_result("gemini_fallback"))).generate_prompt("prompt")
    assert result.provider_error is None
    assert result.attempted_backend_error == "native wrapper failed"
