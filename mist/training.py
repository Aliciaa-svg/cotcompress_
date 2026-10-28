from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _pad_id(model) -> int:
    config = getattr(getattr(model, "base_model", model), "config", None)
    for attr in ("pad_token_id", "eos_token_id"):
        val = getattr(config, attr, None)
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        if val is not None:
            return int(val)
    return 0


def _collate(examples: List[Dict], pad_id: int, max_seq_len: int) -> Optional[Dict[str, torch.Tensor]]:
    rows = []
    for ex in examples:
        ids = list(ex["input_ids"])[:max_seq_len]
        n_masked = min(ex["prompt_len"], len(ids))
        labels = [-100] * n_masked + ids[n_masked:]
        if any(t != -100 for t in labels):
            rows.append((ids, labels))
    if not rows:
        return None
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for b, (ids, lab) in enumerate(rows):
        input_ids[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[b, :len(ids)] = torch.tensor(lab, dtype=torch.long)
        attention_mask[b, :len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def train_lora(
    model,
    examples: List[Dict],
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    learning_rate: float = 5e-5,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    warmup_ratio: float = 0.1,
    max_seq_len: int = 2048,
    seed: int = 42,
):
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    ))
    device = next(peft_model.parameters()).device
    pad_id = _pad_id(peft_model)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)

    steps_per_epoch = max(1, math.ceil(len(examples) / batch_size))
    total_steps = max(1, math.ceil(steps_per_epoch * epochs / grad_accum))
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rng = random.Random(seed)
    data = list(examples)

    peft_model.train()
    optimizer.zero_grad()
    n_micro, running = 0, 0.0
    for epoch in range(epochs):
        rng.shuffle(data)
        bar = tqdm(range(0, len(data), batch_size), total=steps_per_epoch,
                   desc=f"epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for start in bar:
            batch = _collate(data[start:start + batch_size], pad_id, max_seq_len)
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = peft_model(**batch).loss
            (loss / grad_accum).backward()
            running += loss.item()
            n_micro += 1
            if n_micro % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            bar.set_postfix(loss=f"{running / n_micro:.4f}", refresh=False)
    if n_micro % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()
    peft_model.eval()
    return peft_model
