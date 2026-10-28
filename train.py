import argparse

from mist.training import train_lora
from mist.utils import load_model, read_jsonl, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tuning on MIST-compressed chains.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="JSONL written by compress.py")
    ap.add_argument("--output", required=True, help="adapter directory")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    model, _ = load_model(args.model, args.dtype, attn_implementation="sdpa")
    examples = read_jsonl(args.data)
    peft_model = train_lora(
        model, examples,
        rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        learning_rate=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, warmup_ratio=args.warmup_ratio,
        max_seq_len=args.max_seq_len, seed=args.seed,
    )
    peft_model.save_pretrained(args.output)
    print(f"[train] adapter saved to {args.output}")


if __name__ == "__main__":
    main()
