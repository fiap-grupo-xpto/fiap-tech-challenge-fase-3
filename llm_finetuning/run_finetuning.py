"""Script executável equivalente ao notebook fine_tuning_llm_medico.ipynb.

Existe porque a execução headless do notebook via `jupyter nbconvert --execute` no
Windows falha com "Kernel died" logo na primeira célula — um problema conhecido de
incompatibilidade entre o event loop Proactor do asyncio no Windows e o pyzmq usado
pelo kernel do Jupyter, não relacionado à lógica de fine-tuning em si. Este script
roda exatamente o mesmo pipeline (mesmas células, mesma ordem, mesmo código) fora do
kernel Jupyter, produzindo os mesmos artefatos e resultados reais.

O notebook continua sendo a documentação/exploração de referência; este script é o
que efetivamente gera os artefatos de produção.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_finetuning.context_examples import build_context_examples
from backend.assistant.prompts import ASSISTANT_SYSTEM_PROMPT

import html
import json
import os
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11

print(f"PyTorch Versão: {torch.__version__}")
print(f"CUDA Disponível (GPU): {torch.cuda.is_available()}")


# ---------------------------------------------------------------------------
# Parsing do dataset (equivalente à célula 3 do notebook)
# ---------------------------------------------------------------------------
def parse_trec_medical_xml(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Arquivo XML não encontrado no caminho: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for q in root.findall("NLM-QUESTION"):
        qid = q.attrib.get("qid", "")

        orig = q.find("Original-Question")
        subject = orig.find("SUBJECT").text if orig is not None and orig.find("SUBJECT") is not None else ""
        message = orig.find("MESSAGE").text if orig is not None and orig.find("MESSAGE") is not None else ""

        paraphrase = q.find("NIST-PARAPHRASE").text if q.find("NIST-PARAPHRASE") is not None else ""
        summary = q.find("NLM-Summary").text if q.find("NLM-Summary") is not None else ""

        focus_list = []
        qtype = "OUTROS"
        ann = q.find("ANNOTATIONS")
        if ann is not None:
            for f in ann.findall("FOCUS"):
                if f.text:
                    focus_list.append(f.text.strip())
            t_elem = ann.find("TYPE")
            if t_elem is not None and t_elem.text:
                qtype = t_elem.text.strip()

        ref_answers = q.find("ReferenceAnswers")
        if ref_answers is not None:
            for child in ref_answers:
                if "Answer" in child.tag:
                    ans_elem = child.find("ANSWER")
                    ans_text = ans_elem.text if ans_elem is not None else ""
                    url_elem = child.find("AnswerURL")
                    ans_url = url_elem.text if url_elem is not None else ""

                    records.append(
                        {
                            "qid": qid,
                            "subject": subject,
                            "message": message,
                            "paraphrase": paraphrase,
                            "summary": summary,
                            "focus": ", ".join(focus_list),
                            "question_type": qtype,
                            "answer": ans_text,
                            "answer_url": ans_url,
                        }
                    )

    return pd.DataFrame(records)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
RUN_DIR = os.getenv("FINETUNE_RUN_DIR", PROJECT_DIR)
PROCESSED_DATA_DIR = os.path.join(RUN_DIR, "data", "processed")
ARTIFACTS_DIR = os.path.join(RUN_DIR, "artifacts")
EVAL_FIGURES_DIR = os.path.join(RUN_DIR, "evaluation", "figures")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(EVAL_FIGURES_DIR, exist_ok=True)

xml_path = os.path.join(RAW_DATA_DIR, "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml")
df_raw = parse_trec_medical_xml(xml_path)

print(f"Total de pares (Pergunta - Resposta) extraídos do XML: {len(df_raw)}")
print(f"Total de perguntas médicas únicas: {df_raw['qid'].nunique()}")


# ---------------------------------------------------------------------------
# Limpeza (célula 5)
# ---------------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str) or not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df_processed = df_raw.copy()
df_processed["subject_clean"] = df_processed["subject"].apply(clean_text)
df_processed["message_clean"] = df_processed["message"].apply(clean_text)
df_processed["paraphrase_clean"] = df_processed["paraphrase"].apply(clean_text)
df_processed["summary_clean"] = df_processed["summary"].apply(clean_text)
df_processed["answer_clean"] = df_processed["answer"].apply(clean_text)


def build_final_question(row):
    if row["summary_clean"]:
        return row["summary_clean"]
    elif row["paraphrase_clean"]:
        return row["paraphrase_clean"]
    elif row["subject_clean"] and row["message_clean"]:
        return f"{row['subject_clean']}: {row['message_clean']}"
    else:
        return row["message_clean"] or row["subject_clean"]


df_processed["question_clean"] = df_processed.apply(build_final_question, axis=1)

print("Exemplo de Pergunta e Resposta Pré-processadas:")
print("PERGUNTA:", df_processed["question_clean"].iloc[0])
print("RESPOSTA:", df_processed["answer_clean"].iloc[0][:300], "...")


# ---------------------------------------------------------------------------
# Anonimização (célula 7)
# ---------------------------------------------------------------------------
def anonymize_text(text):
    if not isinstance(text, str) or not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"\bNDC#?\s*[\d-]+\b", "[CODIGO_MEDICAMENTO]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b", "[TELEFONE]", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATA]", text)
    text = re.sub(r"\b(Dr\.\s+[A-Z][a-z]+|Dr\s+[A-Z][a-z]+|Doctor\s+[A-Z][a-z]+)\b", "[MEDICO]", text)
    return text.strip()


df_processed["question_anon"] = df_processed["question_clean"].apply(anonymize_text)
df_processed["answer_anon"] = df_processed["answer_clean"].apply(anonymize_text)

mask_changed = (df_processed["question_clean"] != df_processed["question_anon"]) | (
    df_processed["answer_clean"] != df_processed["answer_anon"]
)
print(f"Total de registros com alteração por anonimização: {mask_changed.sum()}")


# ---------------------------------------------------------------------------
# Curadoria (célula 9)
# ---------------------------------------------------------------------------
df_processed["q_words"] = df_processed["question_anon"].apply(lambda x: len(x.split()))
df_processed["a_words"] = df_processed["answer_anon"].apply(lambda x: len(x.split()))

MIN_A_WORDS = 10
MIN_Q_WORDS = 3

df_curated = df_processed[
    (df_processed["a_words"] >= MIN_A_WORDS) & (df_processed["q_words"] >= MIN_Q_WORDS)
].copy()
df_curated = df_curated.drop_duplicates(subset=["question_anon", "answer_anon"])

print("MÉTRICAS DE CURADORIA DE DADOS:")
print(f"   - Registros Brutos Iniciais: {len(df_processed)}")
print(f"   - Registros Curados Finais:  {len(df_curated)}")
print(
    f"   - Amostras Descartadas:      {len(df_processed) - len(df_curated)} "
    f"({((len(df_processed) - len(df_curated)) / len(df_processed)) * 100:.2f}%)"
)


# ---------------------------------------------------------------------------
# EDA (célula 11)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_curated["q_words"], kde=True, color="#2b5c8f", ax=axes[0])
axes[0].set_title("Comprimento das Perguntas (Palavras)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Número de Palavras")
axes[0].set_ylabel("Frequência")

sns.histplot(df_curated["a_words"], kde=True, color="#d95f02", ax=axes[1])
axes[1].set_title("Comprimento das Respostas (Palavras)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Número de Palavras")
axes[1].set_ylabel("Frequência")

top_categories = df_curated["question_type"].value_counts().head(8)
sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis", ax=axes[2])
axes[2].set_title("Top Categorias de Dúvidas Médicas", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Quantidade de Exemplo")

plt.tight_layout()
plt.savefig(os.path.join(EVAL_FIGURES_DIR, "eda_distribuicao_medica.png"), dpi=200)
plt.close(fig)

print("Resumo Estatístico de Comprimento (Palavras):")
print(df_curated[["q_words", "a_words"]].describe().T)


# ---------------------------------------------------------------------------
# Ampliação com protocolos internos do hospital
# ---------------------------------------------------------------------------
# O dataset TREC é QA público de pacientes — não cobre "protocolos médicos do
# hospital" nem "dúvidas de médicos", que o desafio pede como fonte de dados.
# Os pares abaixo são baseados nos MESMOS 5 protocolos usados em tempo de execução
# por backend/data/bootstrap_hospital_db.py (copiados aqui, não importados, para não
# acoplar llm_finetuning ao pacote backend) — fraseados como dúvidas de médicos sobre
# a conduta definida em cada protocolo.
HOSPITAL_PROTOCOL_QA_PAIRS = [
    (
        "PROTO-TRIAGE-001",
        "A patient has persistent cough and a smoking history. What does the pulmonary "
        "triage protocol recommend before concluding next steps?",
        "According to the pulmonary triage protocol (PROTO-TRIAGE-001), when persistent "
        "respiratory symptoms coexist with a smoking history or hemoptysis, the clinician "
        "should prioritize a clinician review and ensure any pending imaging is completed "
        "before concluding next steps.",
    ),
    (
        "PROTO-TRIAGE-001",
        "Why does hemoptysis change the triage priority for a respiratory patient?",
        "Per the pulmonary triage protocol (PROTO-TRIAGE-001), hemoptysis combined with "
        "respiratory symptoms is treated as a trigger for prioritized clinician review and "
        "completion of pending imaging, rather than routine follow-up.",
    ),
    (
        "PROTO-EXAMS-001",
        "What should be checked when a patient has a pending chest CT?",
        "Following the pending exams protocol (PROTO-EXAMS-001), verify the exam status, "
        "the expected completion date, and whether escalation is required; communicate the "
        "pending exam as an alert to the medical team.",
    ),
    (
        "PROTO-EXAMS-001",
        "How should the care team handle a delayed spirometry result?",
        "The pending exams protocol (PROTO-EXAMS-001) requires checking status and expected "
        "completion date, and flagging the pending exam as an alert for the medical team so "
        "it is not overlooked.",
    ),
    (
        "PROTO-ESC-001",
        "When should a respiratory case be escalated for specialized assessment?",
        "According to the escalation protocol (PROTO-ESC-001), high-risk indicators such as "
        "hemoptysis or a high-risk triage classification should trigger immediate clinician "
        "review and escalation to specialized assessment, based on internal triage criteria.",
    ),
    (
        "PROTO-ESC-001",
        "Is it appropriate to state a suspected malignancy diagnosis during escalation?",
        "No. The escalation protocol (PROTO-ESC-001) requires avoiding definitive language "
        "and never claiming malignancy; escalate for specialized clinician assessment instead "
        "of stating a diagnosis.",
    ),
    (
        "PROTO-HITL-001",
        "Can the assistant finalize a treatment decision on its own?",
        "No. Per the human validation protocol (PROTO-HITL-001), all assistant suggestions "
        "require clinician validation; the assistant supports decision-making but must not "
        "replace professional judgment or provide a final diagnosis.",
    ),
    (
        "PROTO-HITL-001",
        "Does a low-risk triage result mean clinician review can be skipped?",
        "No. The human validation protocol (PROTO-HITL-001) requires clinician validation "
        "for all assistant suggestions regardless of the estimated risk level.",
    ),
    (
        "PROTO-NO-RX-001",
        "Can the assistant recommend a specific medication and dosage for a patient?",
        "No. The no-direct-prescription protocol (PROTO-NO-RX-001) states that the assistant "
        "must never prescribe directly or provide dosage instructions; it may at most suggest "
        "that a clinician consider reviewing treatment options per protocol.",
    ),
    (
        "PROTO-NO-RX-001",
        "What should the assistant say if asked to prescribe an antibiotic?",
        "Per the no-direct-prescription protocol (PROTO-NO-RX-001), the assistant should "
        "decline to prescribe or give dosage instructions and instead recommend that a "
        "clinician review appropriate treatment options.",
    ),
]

protocol_rows = [
    {
        "qid": f"PROTO-AUG-{i:02d}",
        "question_type": "protocolo_interno",
        "question_anon": question,
        "answer_anon": answer,
    }
    for i, (_protocol_id, question, answer) in enumerate(HOSPITAL_PROTOCOL_QA_PAIRS, start=1)
]
df_curated = pd.concat([df_curated, pd.DataFrame(protocol_rows), pd.DataFrame(build_context_examples())], ignore_index=True)
print(f"Dataset ampliado com {len(protocol_rows)} exemplos sintéticos de protocolos internos.")
print(f"Total curado + ampliado: {len(df_curated)} amostras.")


# ---------------------------------------------------------------------------
# Tokenizer + formatação com chat template + split agrupado por qid (célula 13)
# ---------------------------------------------------------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = os.path.join(ARTIFACTS_DIR, "llama_medical_lora_model")

SYSTEM_PROMPT = ASSISTANT_SYSTEM_PROMPT

print(f"Carregando Tokenizador para: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def format_llama_instruction(row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question_anon"] if row["question_anon"].startswith("Pergunta clínica:") else f"Pergunta Médica: {row['question_anon']}"},
        {"role": "assistant", "content": row["answer_anon"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


df_curated["text"] = df_curated.apply(format_llama_instruction, axis=1)
MAX_SEQ_LENGTH = int(os.getenv("FINETUNE_MAX_SEQ_LENGTH", "1024"))
# 1024 foi escolhido medindo a distribuição real de tokens do dataset combinado: os 60
# exemplos de context_examples.py ficam todos entre 760-772 tokens (nenhum excluído), os
# 10 exemplos de protocolo ficam entre 177-220 (nenhum excluído), e apenas 3 dos 166
# exemplos curados do TREC excedem 1024 tokens (contra 0 excluídos em 2048). Ou seja,
# 1024 preserva praticamente todo o dataset ampliado pela metade do custo de
# memória/tempo de 2048 — 2048 causou falha de alocação de memória e, na tentativa
# seguinte com bfloat16, um treino tão lento que não terminou nem 1 passo em minutos.
df_curated["token_count"] = df_curated["text"].map(
    lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
)
excluded = df_curated[df_curated["token_count"] > MAX_SEQ_LENGTH]
excluded.to_csv(os.path.join(PROCESSED_DATA_DIR, "excluded_overlength.csv"), index=False)
df_curated = df_curated[df_curated["token_count"] <= MAX_SEQ_LENGTH].copy()
print(f"Excluídas por exceder {MAX_SEQ_LENGTH} tokens (sem truncar respostas): {len(excluded)}")

gss_train_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, val_test_idx = next(gss_train_test.split(df_curated, groups=df_curated["qid"]))
train_df = df_curated.iloc[train_idx]
val_test_df = df_curated.iloc[val_test_idx]

gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss_val_test.split(val_test_df, groups=val_test_df["qid"]))
val_df = val_test_df.iloc[val_idx]
test_df = val_test_df.iloc[test_idx]

assert set(train_df["qid"]).isdisjoint(set(val_df["qid"]))
assert set(train_df["qid"]).isdisjoint(set(test_df["qid"]))
assert set(val_df["qid"]).isdisjoint(set(test_df["qid"]))

# Exporta o dataset processado/formatado — antes o script criava data/processed/ mas
# nunca salvava nada nela.
_export_columns = ["qid", "question_anon", "answer_anon", "text"]
train_df[_export_columns].to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
val_df[_export_columns].to_csv(os.path.join(PROCESSED_DATA_DIR, "validation.csv"), index=False)
test_df[_export_columns].to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)
print(f"Dataset processado (train/validation/test) exportado em: {PROCESSED_DATA_DIR}")

dataset_dict = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df[["question_anon", "answer_anon", "text"]].reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df[["question_anon", "answer_anon", "text"]].reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df[["question_anon", "answer_anon", "text"]].reset_index(drop=True)),
    }
)

print("DIVISÃO DO DATASET (agrupada por qid, sem vazamento entre splits):")
print(f"   - Treino:     {len(dataset_dict['train'])} amostras")
print(f"   - Validação:  {len(dataset_dict['validation'])} amostras")
print(f"   - Teste:      {len(dataset_dict['test'])} amostras")
print("\nExemplo de texto formatado (chat template nativo do tokenizer):")
print(dataset_dict["train"][0]["text"][:400], "...")


# ---------------------------------------------------------------------------
# LoRA + modelo base (célula 15)
# ---------------------------------------------------------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print("Carregando Modelo Causal LLM base...")
device_map = "auto" if torch.cuda.is_available() else None
training_dtype = getattr(torch, os.getenv("FINETUNE_DTYPE", "float16" if torch.cuda.is_available() else "float32"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=training_dtype,
    device_map=device_map,
)
model = get_peft_model(model, peft_config)
model.config.use_cache = False
model.enable_input_require_grads()
print("Resumo de Parâmetros Treináveis LoRA:")
model.print_trainable_parameters()


# ---------------------------------------------------------------------------
# Treinamento real (célula 17)
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = int(os.getenv("FINETUNE_MAX_SEQ_LENGTH", "1024"))  # ver justificativa acima
TRAIN_EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "1"))


def tokenize_function(examples):
    # Não define "labels" aqui: com padding dinâmico (tamanhos variáveis por exemplo),
    # é o DataCollatorForLanguageModeling(mlm=False) quem deve gerar "labels" a partir
    # do input_ids JÁ com padding aplicado — um "labels" pré-definido com tamanhos
    # variados quebra o padding automático do collator (tokenizer.pad() não sabe
    # redimensionar um campo extra que ele não reconhece).
    encoded = tokenizer(examples["text"], truncation=False, add_special_tokens=False)
    if any(len(ids) > MAX_SEQ_LENGTH for ids in encoded["input_ids"]):
        raise ValueError("Training sample exceeds context window; curate it instead of silently truncating the answer")
    return encoded


print("Tokenizando os conjuntos de Treino, Validação e Teste...")
tokenized_datasets = dataset_dict.map(
    tokenize_function, batched=True, remove_columns=dataset_dict["train"].column_names
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    prediction_loss_only=True,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=TRAIN_EPOCHS,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Inicializando o Treinamento Fine-Tuning do modelo médico com LoRA...")
train_result = trainer.train()
trainer.save_state()
trainer.save_metrics("train", train_result.metrics)
print("Fine-tuning concluído com sucesso!")
print(f"Loss final de treino: {train_result.training_loss:.4f}")

with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "base_model_id": MODEL_ID,
            "train_epochs": TRAIN_EPOCHS,
            "dtype": str(training_dtype),
            "seed": SEED,
            "max_seq_length": MAX_SEQ_LENGTH,
            "excluded_overlength_samples": len(excluded),
            "train_samples": len(dataset_dict["train"]),
            "validation_samples": len(dataset_dict["validation"]),
            "test_samples": len(dataset_dict["test"]),
            "final_training_loss": train_result.training_loss,
        },
        f,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(os.path.join(ARTIFACTS_DIR, "tokenizer"))
print(f"Modelo LoRA salvo em: {OUTPUT_DIR}")
model.eval()
model.config.use_cache = True


# ---------------------------------------------------------------------------
# Geração de exemplo base vs fine-tuned (célula 19)
# ---------------------------------------------------------------------------
def generate_medical_response(prompt_question, target_model, max_new_tokens=150):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_question if prompt_question.startswith("Pergunta clínica:") else f"Pergunta Médica: {prompt_question}"},
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_input, return_tensors="pt", add_special_tokens=False).to(target_model.device)
    with torch.no_grad():
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generation = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return generation.strip()


for i in range(min(2, len(dataset_dict["test"]))):
    sample = dataset_dict["test"][i]
    q_test = sample["question_anon"]
    a_ref = sample["answer_anon"]

    print(f"\n{'=' * 70}")
    print(f"TESTE #{i + 1} - PERGUNTA MÉDICA (ANONIMIZADA):")
    print(q_test)
    print("\nRESPOSTA DE REFERÊNCIA (GROUND TRUTH):")
    print(a_ref[:250], "...")

    with model.disable_adapter():
        generated_base = generate_medical_response(q_test, model)
    print("\nRESPOSTA DO MODELO BASE (sem fine-tuning):")
    print(generated_base)

    generated_ans = generate_medical_response(q_test, model)
    print("\nRESPOSTA GERADA PELO MODELO FINE-TUNED (LoRA):")
    print(generated_ans)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Avaliação comparativa real (célula 21)
# ---------------------------------------------------------------------------
def calculate_simple_overlap(reference, candidate):
    ref_tokens = set(reference.lower().split())
    cand_tokens = set(candidate.lower().split())
    if not ref_tokens or not cand_tokens:
        return 0.0, 0.0, 0.0
    intersection = ref_tokens.intersection(cand_tokens)
    precision = len(intersection) / len(cand_tokens)
    recall = len(intersection) / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


eval_results = []
for sample in dataset_dict["test"]:
    q = sample["question_anon"]
    ref = sample["answer_anon"]

    gen_finetuned = generate_medical_response(q, model, max_new_tokens=80)
    with model.disable_adapter():
        gen_base = generate_medical_response(q, model, max_new_tokens=80)

    p_ft, r_ft, f1_ft = calculate_simple_overlap(ref, gen_finetuned)
    p_base, r_base, f1_base = calculate_simple_overlap(ref, gen_base)

    eval_results.append(
        {
            "question": q,
            "reference_answer": ref,
            "base_answer": gen_base,
            "finetuned_answer": gen_finetuned,
            "precision_base": p_base,
            "recall_base": r_base,
            "f1_base": f1_base,
            "precision_finetuned": p_ft,
            "recall_finetuned": r_ft,
            "f1_finetuned": f1_ft,
        }
    )

df_metrics = pd.DataFrame(eval_results)
print("\nDESEMPENHO MÉDIO NO CONJUNTO DE TESTE (overlap de unigramas vs. referência):")
print(
    f"   - Modelo BASE       -> F1 médio: {df_metrics['f1_base'].mean():.4f} "
    f"(precisão {df_metrics['precision_base'].mean():.4f}, revocação {df_metrics['recall_base'].mean():.4f})"
)
print(
    f"   - Modelo FINE-TUNED -> F1 médio: {df_metrics['f1_finetuned'].mean():.4f} "
    f"(precisão {df_metrics['precision_finetuned'].mean():.4f}, revocação {df_metrics['recall_finetuned'].mean():.4f})"
)

results_path = os.path.join(RUN_DIR, "evaluation", "comparative_evaluation_results.csv")
df_metrics.to_csv(results_path, index=False)
print(f"\nResultados comparativos salvos em: {results_path}")
print("\nSCRIPT CONCLUÍDO COM SUCESSO.")
