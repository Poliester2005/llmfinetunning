"""Avalia o modelo base ou o fine-tuned (adaptador LoRA) no conjunto de teste."""
import argparse
import csv
import json

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score

from comum import LABELS, build_messages, load_model, load_tokenizer, parse_label, require_gpu


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def predict(model, tok, texts, bs=16, max_new_tokens=6):
    tok.padding_side = "left"
    preds, raws = [], []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        prompts = [
            tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
            for t in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
        decoded = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws += decoded
        preds += [parse_label(d) for d in decoded]
    return preds, raws


def run_eval(model, tok, rows, bs=16):
    y_true = [r["label"] for r in rows]
    preds, raws = predict(model, tok, [r["text"] for r in rows], bs=bs)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0)
    return acc, f1, preds, raws


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None, help="pasta do adaptador LoRA (omita para avaliar o base)")
    p.add_argument("--split", default="test")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--qlora", action="store_true")
    a = p.parse_args()

    print("GPU:", require_gpu())
    tok = load_tokenizer(a.model)
    model = load_model(a.model, qlora=a.qlora)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    rows = read_jsonl(f"data/{a.split}.jsonl")
    acc, f1, preds, raws = run_eval(model, tok, rows, bs=a.bs)
    y_true = [r["label"] for r in rows]

    tag = "finetuned" if a.adapter else "base"
    print(f"\n=== {tag} | {a.split} | n={len(rows)} ===")
    print(f"Acurácia: {acc:.4f} | F1 macro: {f1:.4f}")
    print(classification_report(y_true, preds, labels=LABELS + ["invalido"], zero_division=0))

    out = f"results/predicoes_{tag}_{a.split}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texto", "esperado", "previsto", "saida_bruta"])
        for r, pr, raw in zip(rows, preds, raws):
            w.writerow([r["text"], r["label"], pr, raw.strip()])
    print("Predições salvas em", out)


if __name__ == "__main__":
    main()
