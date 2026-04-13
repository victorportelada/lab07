"""
Passos 2-4: Fine-tuning QLoRA de um LLM com PEFT/LoRA via SFTTrainer.
Modelo base: NousResearch/Llama-2-7b-hf (requer aceite de licença no HuggingFace).
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------------
BASE_MODEL = "NousResearch/Llama-2-7b-hf"
OUTPUT_DIR = "./results"
ADAPTER_DIR = "./lora-adapter"
TRAIN_FILE = "dataset/train.jsonl"
TEST_FILE = "dataset/test.jsonl"


# ---------------------------------------------------------------------------
# Passo 2: Configuração da Quantização (QLoRA — 4-bit NF4)
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# ---------------------------------------------------------------------------
# Carregamento do modelo base e tokenizer
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------------------------------------------------------------------------
# Passo 3: Arquitetura LoRA
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,           # Rank — dimensão das matrizes de decomposição
    lora_alpha=16,  # Alpha — fator de escala dos novos pesos
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
)

# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------
def format_prompt(example: dict) -> dict:
    """Formata cada par instrução/resposta no template Alpaca."""
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
    )
    return {"text": text}


train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
test_dataset = load_dataset("json", data_files=TEST_FILE, split="train")

train_dataset = train_dataset.map(format_prompt)
test_dataset = test_dataset.map(format_prompt)

# ---------------------------------------------------------------------------
# Passo 4: Configuração do treinamento e otimizador
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",   # paginado p/ CPU — evita OOM na GPU
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,           # Primeiros 3% do treino com warm-up gradual
    lr_scheduler_type="cosine",  # Taxa de aprendizado segue curva cosseno
    report_to="tensorboard",
    evaluation_strategy="epoch",
)

# ---------------------------------------------------------------------------
# SFTTrainer (Supervised Fine-Tuning Trainer — biblioteca trl)
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------------------
# Salvar o adaptador LoRA (apenas os pesos delta, não o modelo completo)
# ---------------------------------------------------------------------------
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adaptador LoRA salvo em: {ADAPTER_DIR}")
