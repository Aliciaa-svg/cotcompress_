import argparse
import pickle
from collections import Counter

from transformers import AutoTokenizer

from mist.compression import BUDGET_POOL, build_sft_examples
from mist.utils import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the MIST-compressed fine-tuning set.")
    ap.add_argument("--model", required=True, help="model whose tokenizer was used for scoring")
    ap.add_argument("--scores", required=True, help="pickle written by score.py")
    ap.add_argument("--output", required=True, help="output JSONL")
    ap.add_argument("--sufficiency", default="pos", choices=["max", "pos"])
    ap.add_argument("--filter-format-tokens", action="store_true",
                    help="exclude whitespace / Markdown / standalone punctuation tokens "
                         "from the retention pool")
    ap.add_argument("--budget-pool", type=float, nargs="+", default=list(BUDGET_POOL))
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.scores, "rb") as f:
        records = pickle.load(f)["records"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    examples = build_sft_examples(
        tokenizer, records,
        sufficiency=args.sufficiency,
        budget_pool=args.budget_pool,
        filter_format_tokens=args.filter_format_tokens,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    write_jsonl(args.output, examples)
    budgets = Counter(ex["gamma"] for ex in examples)
    print(f"[compress] {len(examples)} examples -> {args.output}  budgets: "
          + ", ".join(f"{g}: {n}" for g, n in sorted(budgets.items())))


if __name__ == "__main__":
    main()
