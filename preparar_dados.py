"""Baixa o dataset XLM-T de sentimento em português (tweets) direto do GitHub da Cardiff NLP
e salva em data/*.jsonl (campos: text, label). Não depende da biblioteca `datasets`.

Se o download falhar, crie você mesmo data/train.jsonl, data/validation.jsonl e data/test.jsonl
com uma linha JSON por exemplo: {"text": "...", "label": "negativo|neutro|positivo"}
"""
import json
import pathlib
import urllib.request

BASE = "https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/portuguese"
ID2LABEL = {"0": "negativo", "1": "neutro", "2": "positivo"}
# nome local -> prefixo do arquivo no repositório
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def baixar_linhas(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8").splitlines()


def main():
    pathlib.Path("data").mkdir(exist_ok=True)
    for split, prefix in SPLITS.items():
        textos = baixar_linhas(f"{BASE}/{prefix}_text.txt")
        rotulos = baixar_linhas(f"{BASE}/{prefix}_labels.txt")
        assert len(textos) == len(rotulos), f"{split}: textos e rótulos com tamanhos diferentes"
        path = f"data/{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for t, r in zip(textos, rotulos):
                row = {"text": t.strip(), "label": ID2LABEL[r.strip()]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(textos)} exemplos")


if __name__ == "__main__":
    main()
