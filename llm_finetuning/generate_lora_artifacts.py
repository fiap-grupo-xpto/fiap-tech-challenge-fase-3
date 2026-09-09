import json
import os


def main():
    base_model_id = os.getenv("ITEM1_BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(project_dir, "artifacts")
    model_dir = os.path.join(artifacts_dir, "llama_medical_lora_model")
    tokenizer_dir = os.path.join(artifacts_dir, "tokenizer")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    with open(os.path.join(model_dir, "artifact_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"base_model_id": base_model_id, "type": "peft_lora_adapter"},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Artifacts generated:")
    print(f"- model/adapter: {model_dir}")
    print(f"- tokenizer: {tokenizer_dir}")


if __name__ == "__main__":
    main()

