"""Fine-tuning com LoRA/QLoRA. Cada execução é registrada em results/experimentos.csv."""
import argparse
import csv
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from avaliar import read_jsonl, run_eval
from comum import build_messages, compute_dtype, load_model, load_tokenizer, require_gpu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16, help="rank do LoRA")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--qlora", action="store_true", help="quantização 4-bit (menos VRAM)")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    gpu = require_gpu()
    set_seed(42)
    eff_bs = a.batch_size * a.grad_accum
    name = a.name or f"{a.model.split('/')[-1]}_lr{a.lr}_r{a.r}_ep{a.epochs}_bs{eff_bs}" + ("_q4" if a.qlora else "")
    out_dir = f"outputs/{name}"
    print(f"GPU: {gpu}\nExperimento: {name}")

    # --- dados (formato prompt/completion; a loss é calculada só na resposta) ---
    data = load_dataset(
        "json",
        data_files={"train": "data/train.jsonl", "validation": "data/validation.jsonl"},
    )

    def fmt(ex):
        return {
            "prompt": build_messages(ex["text"]),
            "completion": [{"role": "assistant", "content": ex["label"]}],
        }

    data = data.map(fmt, remove_columns=data["train"].column_names)
    val_rows = read_jsonl("data/validation.jsonl")

    # --- modelo ---
    tok = load_tokenizer(a.model)
    model = load_model(a.model, qlora=a.qlora)
    if a.qlora:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=a.r,
        lora_alpha=2 * a.r,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    bf16 = compute_dtype() == torch.bfloat16
    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_length,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tok,
        peft_config=lora,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    eval_losses = [l["eval_loss"] for l in trainer.state.log_history if "eval_loss" in l]
    eval_loss = eval_losses[-1] if eval_losses else float("nan")

    # --- avaliação por geração na validação (acurácia / F1) ---
    trainer.model.eval()
    acc, f1, _, _ = run_eval(trainer.model, tok, val_rows)

    row = {
        "experimento": name, "modelo": a.model, "lr": a.lr, "lora_r": a.r, "epocas": a.epochs,
        "batch_efetivo": eff_bs, "qlora": a.qlora,
        "train_loss": round(result.training_loss, 4), "eval_loss": round(eval_loss, 4),
        "val_acc": round(acc, 4), "val_f1_macro": round(f1, 4),
        "tempo_min": round(minutes, 2), "pico_vram_gb": round(peak_gb, 2), "gpu": gpu,
    }
    os.makedirs("results", exist_ok=True)
    path = "results/experimentos.csv"
    novo = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if novo:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"Adaptador salvo em {out_dir}")


if __name__ == "__main__":
    main()
