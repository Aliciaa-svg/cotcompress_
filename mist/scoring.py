from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"unsupported architecture: {type(model).__name__}")


def _block_states_and_grads(
    model, input_ids: torch.Tensor, answer_start: int, answer_end: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    layers = get_layers(model)
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    states: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs.retain_grad()
        states.append(hs)

    handles = [layer.register_forward_hook(hook) for layer in layers]
    try:
        logits = model(inputs_embeds=inputs_embeds, use_cache=False).logits
        targets = input_ids[0, answer_start:answer_end]
        log_probs = F.log_softmax(logits[0, answer_start - 1:answer_end - 1].float(), dim=-1)
        log_probs.gather(-1, targets.unsqueeze(-1)).sum().backward()
    finally:
        for h in handles:
            h.remove()

    if len(states) != len(layers):
        raise RuntimeError(f"captured {len(states)} block outputs for {len(layers)} blocks")
    grads = []
    for l, hs in enumerate(states):
        if hs.grad is None:
            raise RuntimeError(f"no gradient at block {l}")
        grads.append(hs.grad)
    return states, grads


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_chain(
    model,
    input_ids: torch.Tensor,
    chain_start: int,
    chain_end: int,
    answer_start: int,
    answer_end: int,
) -> Dict[str, np.ndarray]:
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    assert chain_end > chain_start and answer_end > answer_start >= 1
    n_layers = len(get_layers(model))
    n_chain = chain_end - chain_start

    # --- Full run: prompt + chain + delimiter + answer -------------------
    states, grads = _block_states_and_grads(model, input_ids, answer_start, answer_end)
    h_src = torch.stack(
        [hs[0, chain_start:chain_end].detach().float().cpu() for hs in states]
    )                                                        # [L, T, d]
    g_src = torch.stack(
        [g[0, chain_start:chain_end].detach().float().cpu() for g in grads]
    )
    theta_hat = (g_src * h_src).sum(dim=-1)                  # [L, T]

    with torch.no_grad():
        h_prev = (
            model.get_input_embeddings()(input_ids[0, chain_start:chain_end])
            .detach().float().cpu()
        )
        unembed = model.lm_head.weight
        answer_ids = input_ids[0, answer_start:answer_end].to(unembed.device)
        u = unembed[answer_ids].sum(dim=0).detach().float().cpu()   # [d]
    layer_contrib = torch.zeros(n_layers, n_chain, dtype=torch.float32)
    for l in range(n_layers):
        layer_contrib[l] = (h_src[l] - h_prev) @ u
        h_prev = h_src[l]
    c_bar = layer_contrib.mean(dim=1)                        # [L]
    theta = (c_bar.unsqueeze(1) * theta_hat.abs()).sum(dim=0)

    del states, grads, g_src
    _free_cuda()

    target_ids = torch.cat(
        [input_ids[0, :chain_start], input_ids[0, chain_end:]]
    ).unsqueeze(0)
    tgt_answer_start = answer_start - n_chain
    tgt_answer_end = answer_end - n_chain
    if not torch.equal(
        input_ids[0, answer_start:answer_end],
        target_ids[0, tgt_answer_start:tgt_answer_end],
    ):
        raise RuntimeError("answer span changed after removing the chain")
    states, grads = _block_states_and_grads(
        model, target_ids, tgt_answer_start, tgt_answer_end,
    )
    final = tgt_answer_start - 1
    h_final = torch.stack([hs[0, final].detach().float().cpu() for hs in states])  # [L, d]
    g_final = torch.stack([g[0, final].detach().float().cpu() for g in grads])
    del states, grads
    _free_cuda()

    lambda_hat = torch.einsum(
        "ltd,ld->lt", h_src - h_final.unsqueeze(1), g_final,
    )                                                        # [L, T]
    lam = lambda_hat.max(dim=0).values
    positive = (lambda_hat > 0).to(lambda_hat.dtype)
    lam_pos = torch.where(
        positive.any(dim=0),
        (c_bar.unsqueeze(1) * lambda_hat * positive).sum(dim=0),
        lam,
    )

    for name, t in (("theta", theta), ("lambda", lambda_hat), ("lambda+", lam_pos)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"non-finite {name}")
    if float(theta_hat.abs().sum()) <= 0:
        raise RuntimeError("zero necessity gradient mass")
    g_norm = g_final.norm(dim=-1)
    if not torch.isfinite(g_norm).all() or float(g_norm.min()) == 0.0:
        raise RuntimeError("zero sufficiency gradient at some layer")

    return {
        "theta": theta.numpy().astype(np.float32),
        "lam": lam.numpy().astype(np.float32),
        "lam_pos": lam_pos.numpy().astype(np.float32),
    }


def _log_z(x: np.ndarray) -> np.ndarray:
    log_abs = np.log(np.abs(x) + 1e-12)
    s = float(np.std(log_abs))
    if s < 1e-12:
        return np.zeros_like(log_abs)
    return (log_abs - float(np.mean(log_abs))) / s
def _z(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / s


def mist_score(theta: np.ndarray, lam: np.ndarray, gamma: float) -> np.ndarray:
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    lam = np.asarray(lam, dtype=np.float64).reshape(-1)
    if theta.shape != lam.shape:
        raise ValueError(f"shape mismatch: {theta.shape} vs {lam.shape}")
    return (gamma * _log_z(theta) + (1.0 - gamma) * _z(lam)).astype(np.float32)


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(scores)))
    return np.sort(np.argpartition(scores, -k)[-k:])
