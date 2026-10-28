from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_model(name: str, dtype: str = "bfloat16", attn_implementation: str = "sdpa",
               frozen: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs = {"torch_dtype": DTYPES[dtype], "attn_implementation": attn_implementation}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
