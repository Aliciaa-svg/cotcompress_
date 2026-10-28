"""Budget-conditioned greedy decoding and answer checking"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm.auto import tqdm

from .prompts import encode_prompt

# Decoding budget used in the paper.
MAX_NEW_TOKENS = {"gsm8k": 512, "math": 1024}


def extract_answer(text: str, dataset: str) -> Optional[Union[float, str]]:
    if dataset == "math":
        from .math_grader import extract_math_answer
        matches = extract_math_answer("", text)
        return matches[-1] if matches else None

    cleaned = text.replace(",", "")
    boxed = list(re.finditer(r"\\boxed\{([^{}]+)\}", cleaned))
    if boxed:
        nums = re.findall(r"-?\d+\.?\d*", boxed[-1].group(1).strip())
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                pass
    for pattern in (r"####\s*(-?\d+\.?\d*)",
                    r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s\$\\]*(-?\d+\.?\d*)"):
        m = re.search(pattern, cleaned)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    nums = re.findall(r"-?\d+\.?\d*", cleaned)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def check_answer(pred: Optional[Union[float, str]], gold: str, dataset: str,
                 tol: float = 1e-3) -> bool:
    if pred is None:
        return False
    if dataset == "math":
        if not isinstance(pred, str):
            return False
        from .math_grader import math_equal, strip_string
        gold_norm = strip_string(gold or "")
        if not gold_norm:
            return False
        return bool(math_equal(strip_string(pred), gold_norm, timeout=True))
    if not isinstance(pred, (int, float)):
        return False
    try:
        g = float(str(gold).replace(",", "").strip())
    except (TypeError, ValueError):
        return False
    if g == 0:
        return abs(pred) < tol
    return abs(pred - g) / (abs(g) + 1e-12) < tol


def _left_pad(prompts: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for b, p in enumerate(prompts):
        input_ids[b, width - len(p):] = torch.tensor(p, dtype=torch.long)
        attention_mask[b, width - len(p):] = 1
    return input_ids, attention_mask, width


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    examples: List[Dict[str, str]],
    gamma: float,
    dataset: str,
    max_new_tokens: int,
    batch_size: int = 16,
) -> Tuple[Dict, List[Dict]]:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    prompts = [encode_prompt(tokenizer, ex["question"], gamma) for ex in examples]
    records: List[Dict] = []
    seconds = 0.0
    for start in tqdm(range(0, len(prompts), batch_size), desc=f"gamma={gamma}",
                      dynamic_ncols=True):
        input_ids, attention_mask, width = _left_pad(prompts[start:start + batch_size], pad_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Neutralize sampling defaults shipped in some generation configs.
            temperature=1.0, top_p=1.0, top_k=0,
            pad_token_id=pad_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        for b, ex in enumerate(examples[start:start + batch_size]):
            gen = out[b, width:]
            n_tokens = int(gen.size(0))
            if eos_id is not None:
                eos_pos = (gen == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    n_tokens = int(eos_pos[0, 0].item()) + 1
            text = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_answer(text, dataset)
            records.append({
                "qid": start + b,
                "gamma": gamma,
                "correct": check_answer(pred, ex["answer"], dataset),
                "pred": pred,
                "gold": ex["answer"],
                "output_tokens": n_tokens,
                "text": text,
            })
    n = max(1, len(records))
    summary = {
        "accuracy": sum(r["correct"] for r in records) / n,
        "correct": sum(r["correct"] for r in records),
        "total": len(records),
        "mean_output_tokens": sum(r["output_tokens"] for r in records) / n,
        "seconds": seconds,
    }
    return summary, records
