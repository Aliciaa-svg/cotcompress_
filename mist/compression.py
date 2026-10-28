"""Budgeted chain compression and fine-tuning data."""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .prompts import ANSWER_SUFFIX, encode_prompt
from .scoring import mist_score, select_top_k

BUDGET_POOL = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SUFFICIENCY_KEYS = {"max": "lam", "pos": "lam_pos"}

_KEEP_CHARS = set("=+/(){}[]<>$\\^_|!%&~`'\"")
_WORD_CHAR = re.compile(r"\w", flags=re.UNICODE)
_FORMAT_PATTERNS = [
    re.compile(p) for p in (
        r"^\s+$",                       # whitespace
        r"^\*{2,}$",                    # ** ***
        r"^#+$",                        # # ## ###
        r"^-{2,}$",                     # -- ---
        r"^-{2,}\s*\n+\s*$",
        r"^\s*\n+\s*-{2,}\s*$",
        r"^[:;,]+$",                    # standalone : ; ,  ("." is kept)
        r"^[:;,]+\s+$",
        r"^\s+[:;,]+$",
        r"^[:;,]+\s*\n+\s*$",
        r"^\s*\n+\s*[:;,]+$",
        r"^\*{2,}\s*\n+\s*$",
        r"^\s*\n+\s*\*{2,}\s*$",
        "^[ -⁯⸀-⹿]+$",                                     # punctuation blocks
        "^[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F☀-➿]+$",       # emoji / symbols
    )
]


def is_format_token(text: str) -> bool:
    if not text or _WORD_CHAR.search(text):
        return False
    if any(ch in _KEEP_CHARS for ch in text):
        return False
    return any(p.match(text) for p in _FORMAT_PATTERNS)


def strip_chain_tail(text: str) -> str:
    prev = None
    while text != prev:
        prev = text
        text = text.rstrip()
        if text.endswith("\\(") or text.endswith("\\["):
            text = text[:-2]
            continue
        if text.endswith("$") or text.endswith("\\"):
            text = text[:-1]
            continue
        for w in (" is", " are", " equals", " equal to", " = ", " to"):
            if text.lower().endswith(w):
                text = text[: -len(w)]
                break
        else:
            break
    return text.rstrip()


def select_chain_tokens(
    scores: np.ndarray,
    gamma: float,
    piece_texts: Optional[Sequence[str]] = None,
) -> np.ndarray:
    k = max(1, math.ceil(gamma * len(scores)))
    if piece_texts is not None:
        mask = np.fromiter((is_format_token(t) for t in piece_texts), dtype=bool,
                           count=len(piece_texts))
        n_keep = int((~mask).sum())
        if n_keep > 0:
            scores = scores.copy()
            scores[mask] = -np.inf
            k = min(k, n_keep)
    return select_top_k(scores, k)


def build_sft_examples(
    tokenizer,
    records: List[Dict],
    sufficiency: str = "pos",
    budget_pool: Sequence[float] = BUDGET_POOL,
    filter_format_tokens: bool = False,
    max_seq_len: int = 2048,
    seed: int = 42,
) -> List[Dict]:
    lam_key = SUFFICIENCY_KEYS[sufficiency]
    rng = random.Random(seed)
    eos_id = tokenizer.eos_token_id
    examples: List[Dict] = []
    for rec in records:
        gamma = rng.choice(budget_pool)
        ids = [int(t) for t in rec["input_ids"]]
        cs, ce, as_, ae = rec["spans"]
        chain_ids = ids[cs:ce]

        if gamma >= 1.0:
            kept = list(chain_ids)
        else:
            scores = mist_score(rec["theta"], rec[lam_key], gamma)
            if len(scores) != len(chain_ids):
                continue
            pieces = ([tokenizer.decode([t]) for t in chain_ids]
                      if filter_format_tokens else None)
            kept = [chain_ids[i] for i in select_chain_tokens(scores, gamma, pieces)]

        if kept:
            text = tokenizer.decode(kept, skip_special_tokens=True)
            cleaned = strip_chain_tail(text)
            if cleaned != text:
                kept = tokenizer.encode(cleaned, add_special_tokens=False)

        answer = tokenizer.decode(ids[as_:ae], skip_special_tokens=True).strip()
        target = kept + tokenizer.encode(ANSWER_SUFFIX.format(answer=answer),
                                         add_special_tokens=False)
        if eos_id is not None:
            target.append(eos_id)
        prompt_ids = encode_prompt(tokenizer, rec["question"], gamma)
        examples.append({
            "input_ids": (prompt_ids + target)[:max_seq_len],
            "prompt_len": len(prompt_ids),
            "gamma": gamma,
        })
    rng.shuffle(examples)
    return examples
