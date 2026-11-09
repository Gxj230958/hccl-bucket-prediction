#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cards=${1:?Usage: bash run_scaled.sh CARDS FRESH_OUTPUT_DIRECTORY}
output=${2:?Usage: bash run_scaled.sh CARDS FRESH_OUTPUT_DIRECTORY}
IFS=, read -r -a devices <<< "$cards"
ranks=${#devices[@]}
mkdir -p "$(dirname "$output")"
mkdir "$output"
for condition in none cube; do
    for repeat in 0 1 2; do
        bash scripts/run_npu.sh "$cards" .venv/bin/python -m torch.distributed.run \
            --standalone --nproc-per-node="$ranks" collect.py --microbench \
            --concurrent "$condition" --output "$output/cost_${condition}_${repeat}" \
            > "$output/cost_${condition}_${repeat}.log" 2>&1
    done
done
bash scripts/run_npu.sh "$cards" .venv/bin/python -m torch.distributed.run \
    --standalone --nproc-per-node="$ranks" collect.py --bucket-mb 2048 \
    --sequence 256 --steps 6 --seed 321 --parameter-events \
    --model-config configs/qwen3_06b.json \
    --data data/calibration.json \
    --output "$output/calibration" > "$output/calibration.log" 2>&1
for condition in none cube; do
    .venv/bin/python predict.py --trace "$output"/calibration/train_rank*.json \
        --collectives "$output"/cost_*/collectives_rank*.json --world-size "$ranks" \
        --condition "$condition" --cost-field device_ms --buckets-mb 2 8 25 32 128 \
        --output "$output/prediction_${condition}.json"
done
# The compute-overlapped cost model is fixed before evaluating any candidate.
cp "$output/prediction_cube.json" "$output/prediction.json"
for repeat in 0 1 2; do
    for bucket in 2 8 25 32 128; do
        bash scripts/run_npu.sh "$cards" .venv/bin/python -m torch.distributed.run \
            --standalone --nproc-per-node="$ranks" collect.py --bucket-mb "$bucket" \
            --sequence 256 --steps 6 --seed "$((1000+repeat))" \
            --model-config configs/qwen3_06b.json \
            --data data/evaluation.json \
            --output "$output/eval_${repeat}_${bucket}" \
            > "$output/eval_${repeat}_${bucket}.log" 2>&1
    done
done
.venv/bin/python evaluate.py "$output"
.venv/bin/python evaluate.py "$output" --prediction "$output/prediction_none.json" \
    --output "$output/evaluation_none.json"
