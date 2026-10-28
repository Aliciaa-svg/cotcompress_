from __future__ import annotations

from typing import List, Tuple

import torch

SYSTEM_MESSAGE = "You are a helpful assistant."
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}.\n"
PLAIN_TEMPLATE = "Question: {q}\nAnswer: "
ANSWER_DELIMITER = "\n#### "
RATIO_TAG = "<|eot_id|>"
ANSWER_SUFFIX = "\n\nThe final answer is: $\\boxed{{{answer}}}$"


def build_prompt_text(tokenizer, question: str) -> str:
    content = f"{INSTRUCTION}{question}"
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass
    return PLAIN_TEMPLATE.format(q=content)


def encode_text(tokenizer, text: str) -> List[int]:
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode(text, add_special_tokens=True)


def encode_prompt(tokenizer, question: str, gamma: float) -> List[int]:
    if gamma >= 1.0:
        q = question
    else:
        eos = getattr(tokenizer, "eos_token", None)
        sep = "\n" if (eos is not None and RATIO_TAG == eos) else ""
        q = f"{question}{sep}{RATIO_TAG}{gamma}{RATIO_TAG}"
    return encode_text(tokenizer, build_prompt_text(tokenizer, q))


def build_spans(
    tokenizer, question: str, chain: str, answer: str,
) -> Tuple[torch.Tensor, int, int, int, int]:
    prompt = build_prompt_text(tokenizer, question)
    ids_prompt = encode_text(tokenizer, prompt)
    ids_chain = encode_text(tokenizer, prompt + chain)
    ids_delim = encode_text(tokenizer, prompt + chain + ANSWER_DELIMITER)
    ids_full = encode_text(tokenizer, prompt + chain + ANSWER_DELIMITER + answer)
    if (
        ids_chain[: len(ids_prompt)] == ids_prompt
        and ids_delim[: len(ids_chain)] == ids_chain
        and ids_full[: len(ids_delim)] == ids_delim
    ):
        spans = (len(ids_prompt), len(ids_chain), len(ids_delim), len(ids_full))
        _validate_spans(*spans)
        return (torch.tensor([ids_full], dtype=torch.long),) + spans
    return _spans_via_offsets(tokenizer, prompt, chain, answer)


def _spans_via_offsets(tokenizer, prompt: str, chain: str, answer: str):
    full_text = prompt + chain + ANSWER_DELIMITER + answer
    bos = getattr(tokenizer, "bos_token", None)
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=not (bos and full_text.startswith(bos)),
    )
    offsets = enc["offset_mapping"][0].tolist()
    p_end = len(prompt)
    c_end = p_end + len(chain)
    d_end = c_end + len(ANSWER_DELIMITER)
    chain_start = chain_end = answer_start = None
    for i, (_, end) in enumerate(offsets):
        if chain_start is None and end > p_end:
            chain_start = i
        if chain_end is None and end > c_end:
            chain_end = i
        if answer_start is None and end > d_end:
            answer_start = i
            break
    if chain_start is None:
        raise ValueError("no chain tokens found")
    if chain_end is None:
        chain_end = len(offsets)
    if answer_start is None:
        answer_start = chain_end
    answer_end = len(offsets)
    _validate_spans(chain_start, chain_end, answer_start, answer_end)
    return enc["input_ids"], chain_start, chain_end, answer_start, answer_end


def _validate_spans(chain_start: int, chain_end: int, answer_start: int, answer_end: int) -> None:
    if chain_start < 1:
        raise ValueError(f"chain_start={chain_start} must be >= 1")
    if chain_end <= chain_start:
        raise ValueError(f"empty chain span [{chain_start}, {chain_end})")
    if answer_start < chain_end:
        raise ValueError(f"answer_start={answer_start} < chain_end={chain_end}")
    if answer_end <= answer_start:
        raise ValueError(f"empty answer span [{answer_start}, {answer_end})")
