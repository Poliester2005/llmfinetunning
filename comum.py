"""Funções compartilhadas: prompt, labels, carregamento de modelo."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LABELS = ["negativo", "neutro", "positivo"]
SYSTEM_PROMPT = "Você é um assistente que classifica o sentimento de textos em português."
DEFAULT_INSTRUCTION = (
    "Classifique o sentimento do texto abaixo como negativo, neutro ou positivo. "
    "Responda apenas com uma palavra."
)


def build_messages(text, instruction=DEFAULT_INSTRUCTION):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\nTexto: {text}"},
    ]


def parse_label(generated):
    g = generated.strip().lower()
    for label in LABELS:
        if label in g:
            return label
    return "invalido"


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA não disponível. Instale o PyTorch com suporte a CUDA e confira com nvidia-smi."
        )
    return torch.cuda.get_device_name(0)


def compute_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name, qlora=False):
    kwargs = {"device_map": {"": 0}}  # força tudo na GPU 0
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )
    else:
        kwargs["dtype"] = compute_dtype()
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
