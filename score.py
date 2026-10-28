import argparse
import pickle
from pathlib import Path

import torch
from tqdm.auto import tqdm

from mist.data import load_chains
from mist.prompts import build_spans
from mist.scoring import score_chain
from mist.utils import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Score chain tokens with MIST.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--chains", required=True, help="JSONL with question / chain / answer")
    ap.add_argument("--output", required=True, help="output pickle")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-chain-tokens", type=int, default=1024,
                    help="chains longer than this (or shorter than 3 tokens) are skipped")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N chains")
    args = ap.parse_args()

    chains = load_chains(args.chains, args.limit)
    model, tokenizer = load_model(args.model, args.dtype, attn_implementation="eager", frozen=True)
    device = next(model.parameters()).device

    records = []
    for i, ex in enumerate(tqdm(chains, desc="score", dynamic_ncols=True)):
        try:
            input_ids, cs, ce, as_, ae = build_spans(
                tokenizer, ex["question"], ex["chain"], ex["answer"])
        except Exception as e:
            print(f"[score] chain {i}: cannot tokenize ({e}); skipped")
            continue
        if not 3 <= ce - cs <= args.max_chain_tokens:
            continue
        try:
            scores = score_chain(model, input_ids.to(device), cs, ce, as_, ae)
        except Exception as e:
            print(f"[score] chain {i}: {e}; skipped")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        records.append({
            "index": i,
            "question": ex["question"],
            "input_ids": input_ids[0].tolist(),
            "spans": (cs, ce, as_, ae),
            **scores,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"model": args.model, "dtype": args.dtype, "records": records}, f)
    print(f"[score] {len(records)} / {len(chains)} chains scored -> {args.output}")


if __name__ == "__main__":
    main()
