"""Frontend (Gradio): o usuário escreve o texto e, se quiser, edita a instrução (prompt)."""
import argparse
from contextlib import nullcontext

import gradio as gr
import torch
from peft import PeftModel

from comum import (DEFAULT_INSTRUCTION, build_messages, load_model, load_tokenizer,
                   parse_label, require_gpu)

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--adapter", default=None, help="pasta do adaptador LoRA treinado")
p.add_argument("--qlora", action="store_true")
args = p.parse_args()

gpu = require_gpu()
tok = load_tokenizer(args.model)
model = load_model(args.model, qlora=args.qlora)
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

VERSOES = ["Fine-tuned (LoRA)", "Base"] if args.adapter else ["Base"]


@torch.no_grad()
def gerar(texto, instrucao, versao, temperatura, max_tokens):
    prompt = tok.apply_chat_template(
        build_messages(texto, instrucao), tokenize=False, add_generation_prompt=True
    )
    enc = tok(prompt, return_tensors="pt").to(model.device)
    kw = {"max_new_tokens": int(max_tokens), "pad_token_id": tok.pad_token_id}
    if temperatura > 0:
        kw.update(do_sample=True, temperature=float(temperatura), top_p=0.9)
    else:
        kw.update(do_sample=False)
    usar_base = args.adapter and versao == "Base"
    with (model.disable_adapter() if usar_base else nullcontext()):
        out = model.generate(**enc, **kw)
    resp = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return resp, parse_label(resp)


with gr.Blocks(title="Classificador de sentimento (LLM local)") as demo:
    gr.Markdown(f"## Classificador de sentimento — {args.model}\nGPU: **{gpu}**")
    texto = gr.Textbox(label="Texto", lines=4, placeholder="Digite um texto em português...")
    instrucao = gr.Textbox(label="Instrução (prompt) — editável", value=DEFAULT_INSTRUCTION, lines=3)
    with gr.Row():
        versao = gr.Radio(VERSOES, value=VERSOES[0], label="Modelo")
        temperatura = gr.Slider(0, 1.5, value=0, step=0.1, label="Temperatura (0 = determinístico)")
        max_tokens = gr.Slider(1, 64, value=8, step=1, label="Máx. de tokens")
    botao = gr.Button("Classificar", variant="primary")
    saida = gr.Textbox(label="Resposta do modelo")
    rotulo = gr.Textbox(label="Rótulo interpretado")
    botao.click(gerar, [texto, instrucao, versao, temperatura, max_tokens], [saida, rotulo])

demo.launch()
