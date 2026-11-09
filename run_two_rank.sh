#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
output=${1:?Usage: bash run_two_rank.sh FRESH_OUTPUT_DIRECTORY}
mkdir -p "$(dirname "$output")"
mkdir "$output"
for repeat in 0 1 2; do
    bash scripts/run_npu.sh 12,13 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 collect.py \
        --microbench --output "$output/cost_${repeat}" > "$output/cost_${repeat}.log" 2>&1
done
bash scripts/run_npu.sh 12,13 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 collect.py \
    --bucket-mb 2048 --steps 6 --seed 321 --parameter-events \
    --model-config configs/qwen3_06b.json \
    --output "$output/calibration" > "$output/calibration.log" 2>&1
.venv/bin/python predict.py --trace "$output"/calibration/train_rank*.json \
    --collectives "$output"/cost_*/collectives_rank*.json --world-size 2 \
    --buckets-mb 1 4 16 25 64 256 --output "$output/prediction.json"
for repeat in 0 1 2; do
    for bucket in 1 4 16 25 64 256; do
        bash scripts/run_npu.sh 12,13 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 collect.py \
            --bucket-mb "$bucket" --steps 6 --seed "$((1000+repeat))" \
            --model-config configs/qwen3_06b.json \
            --output "$output/eval_${repeat}_${bucket}" > "$output/eval_${repeat}_${bucket}.log" 2>&1
    done
done
.venv/bin/python evaluate.py "$output"
