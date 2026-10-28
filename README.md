# MIST

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python score.py --model meta-llama/Llama-3.1-8B-Instruct --chains data/math_train.jsonl \
    --output outputs/math/scores.pkl

python compress.py --model meta-llama/Llama-3.1-8B-Instruct --scores outputs/math/scores.pkl \
    --output outputs/math/sft.jsonl

python train.py --model meta-llama/Llama-3.1-8B-Instruct --data outputs/math/sft.jsonl \
    --output outputs/math/adapter

python evaluate.py --model meta-llama/Llama-3.1-8B-Instruct --adapter outputs/math/adapter \
    --dataset math --output outputs/math/eval.json
```

`run.sh` runs all four steps for both sufficiency aggregates.