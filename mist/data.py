"""Evaluation sets and the chain file format."""

from __future__ import annotations

from typing import Dict, List, Optional

from .utils import read_jsonl


def load_test_set(dataset: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    from datasets import load_dataset

    examples: List[Dict[str, str]] = []
    if dataset == "gsm8k":
        for ex in load_dataset("openai/gsm8k", "main", split="test"):
            if "####" not in ex["answer"]:
                continue
            final = ex["answer"].split("####", 1)[1]
            examples.append({"question": ex["question"].strip(), "answer": final.strip()})
    elif dataset == "math":
        from .math_grader import extract_answer

        for ex in load_dataset("HuggingFaceH4/MATH-500", split="test"):
            problem = (ex.get("problem") or "").strip()
            solution = (ex.get("solution") or "").strip()
            if not problem or not solution:
                continue
            answer = (ex.get("answer") or "").strip() or extract_answer(solution)
            if answer:
                examples.append({"question": problem, "answer": answer})
    return examples[:limit] if limit else examples


def load_chains(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    chains = read_jsonl(path)
    for i, row in enumerate(chains):
        missing = [k for k in ("question", "chain", "answer") if not isinstance(row.get(k), str)]
        if missing:
            raise ValueError(f"{path}:{i + 1}: missing string field(s) {missing}")
    return chains[:limit] if limit else chains
