import argparse
import json
from pathlib import Path

from mist.data import DATASETS, load_test_set
from mist.evaluation import MAX_NEW_TOKENS, evaluate
from mist.utils import load_model, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Budget-conditioned evaluation of a MIST adapter.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="directory written by train.py")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--output", required=True, help="output JSON")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: 512 for GSM8K, 1024 for MATH")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    args = ap.parse_args()

    from peft import PeftModel

    max_new_tokens = args.max_new_tokens or MAX_NEW_TOKENS[args.dataset]
    examples = load_test_set(args.dataset, args.limit)
    model, tokenizer = load_model(args.model, args.dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    results, all_records = {}, []
    for gamma in args.gammas:
        summary, records = evaluate(model, tokenizer, examples, gamma, args.dataset,
                                    max_new_tokens, args.batch_size)
        results[str(gamma)] = summary
        all_records.extend(records)
        print(f"[eval] gamma={gamma}: accuracy {100 * summary['accuracy']:.2f}% "
              f"({summary['correct']}/{summary['total']}), "
              f"{summary['mean_output_tokens']:.1f} generated tokens")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "model": args.model, "adapter": args.adapter, "dataset": args.dataset,
            "max_new_tokens": max_new_tokens, "n_questions": len(examples),
            "results": results,
        }, f, indent=2)
    write_jsonl(str(out.with_suffix(".jsonl")), all_records)
    print(f"[eval] results -> {out}")


if __name__ == "__main__":
    main()
