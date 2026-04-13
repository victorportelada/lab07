# Lab 07 — Fine-Tuning de LLM com LoRA e QLoRA

Pipeline de fine-tuning para um modelo de linguagem causal usando PEFT/LoRA e quantização em 4 bits (QLoRA).

## Domínio

Assistente de programação Python — o modelo é especializado para responder perguntas e explicar conceitos de Python com exemplos de código.

## Estrutura do Repositório

```
.
├── generate_dataset.py   # Passo 1: geração de dataset sintético via API da OpenAI
├── train.py              # Passos 2–4: pipeline de treinamento com QLoRA
├── dataset/
│   ├── train.jsonl       # 90% dos dados (gerado por generate_dataset.py)
│   └── test.jsonl        # 10% dos dados
└── lora-adapter/         # Pesos do adaptador LoRA salvos (gerado por train.py)
```

## Configuração

```bash
pip install openai transformers datasets peft trl bitsandbytes accelerate
```

## Uso

### 1. Gerar o dataset sintético

```bash
export OPENAI_API_KEY="sua-chave-aqui"
python generate_dataset.py
```

Gera 60 pares instrução/resposta (20 tópicos de Python × 3 cada), embaralha e salva:
- `dataset/train.jsonl` — 54 exemplos (90%)
- `dataset/test.jsonl` — 6 exemplos (10%)

### 2. Executar o fine-tuning

```bash
python train.py
```

Requer uma GPU CUDA com pelo menos 8 GB de VRAM (testado em T4/A10G).

## Hiperparâmetros Principais

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| Quantização | 4-bit NF4 | Reduz uso de VRAM |
| LoRA rank (r) | 64 | Dimensão da matriz de decomposição |
| LoRA alpha | 16 | Fator de escalonamento dos pesos |
| LoRA dropout | 0.1 | Evita overfitting |
| Otimizador | paged_adamw_32bit | Descarrega picos de memória para a CPU |
| LR scheduler | cosine | Decaimento suave da taxa de aprendizado |
| Warmup ratio | 0.03 | Aquecimento gradual nos primeiros 3% dos passos |

## Uso de IA
A IA foi utilizada apenas para brainstorming e geração de ideias iniciais. Todo o código foi desenvolvido, revisado e validado manualmente.