"""
Passo 1: Geração de dataset sintético via OpenAI API.
Domínio: Assistente de programação Python.
Gera 50+ pares instrução/resposta, divide 90/10 e salva em .jsonl.
"""

import json
import os
import random
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DOMAIN_TOPICS = [
    "list comprehensions",
    "decorators",
    "generators",
    "context managers",
    "async/await",
    "dataclasses",
    "type hints",
    "error handling with try/except",
    "file I/O operations",
    "string formatting",
    "lambda functions",
    "map and filter",
    "itertools",
    "collections module",
    "pathlib",
    "argparse",
    "logging module",
    "unittest",
    "virtual environments",
    "packaging with setuptools",
]

SYSTEM_PROMPT = (
    "You are an expert Python programming instructor. "
    "Generate a realistic and educational Q&A pair about the given Python topic. "
    "Return ONLY a JSON object with two keys: 'instruction' (a question or task) "
    "and 'response' (a clear, concise answer with a code example when relevant). "
    "Do not include any text outside the JSON object."
)


def generate_pair(topic: str) -> dict:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic}"},
        ],
        temperature=0.8,
    )
    raw = completion.choices[0].message.content.strip()
    return json.loads(raw)


def main():
    os.makedirs("dataset", exist_ok=True)

    pairs = []
    # Generate at least 3 pairs per topic to reach 50+ total (20 topics * 3 = 60)
    for topic in DOMAIN_TOPICS:
        for attempt in range(3):
            try:
                pair = generate_pair(topic)
                pairs.append(pair)
                print(f"[OK] {topic} ({len(pairs)}/60)")
            except Exception as exc:
                print(f"[WARN] {topic} attempt {attempt + 1} failed: {exc}")

    print(f"\nTotal pairs generated: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)

    split = int(len(pairs) * 0.9)
    train_data = pairs[:split]
    test_data = pairs[split:]

    def write_jsonl(path: str, records: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_jsonl("dataset/train.jsonl", train_data)
    write_jsonl("dataset/test.jsonl", test_data)

    print(f"Train: {len(train_data)} examples -> dataset/train.jsonl")
    print(f"Test : {len(test_data)} examples  -> dataset/test.jsonl")


if __name__ == "__main__":
    main()
