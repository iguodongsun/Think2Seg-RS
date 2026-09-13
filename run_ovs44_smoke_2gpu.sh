#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/sam3/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/OVS44Reason}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/think2seg-rs/models/Think2Seg-RS-3B}
SAM_ROOT=${SAM_ROOT:-/root/autodl-tmp/think2seg-rs/sam2}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/think2seg-rs/ovs44_smoke}
MAX_SAMPLES=${MAX_SAMPLES:-2}

cd /home/Think2Seg-RS
"$PYTHON" tools/prepare_ovs44.py --root "$DATA_ROOT" --output data/ovs44
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$SAM_ROOT:/home/Think2Seg-RS/src/open-r1-multimodal/src:${PYTHONPATH:-}" \
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/eval/eval_ovs44.py \
  --manifest data/ovs44/test.json \
  --model "$MODEL_PATH" \
  --sam-root "$SAM_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES"
