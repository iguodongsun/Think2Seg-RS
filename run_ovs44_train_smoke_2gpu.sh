#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/sam3/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/OVS44Reason}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/think2seg-rs/models/Think2Seg-RS-3B}
SAM_ROOT=${SAM_ROOT:-/root/autodl-tmp/think2seg-rs/sam2}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/think2seg-rs/ovs44_train_smoke}
GRAD_ACCUM=${GRAD_ACCUM:-2}

cd /home/Think2Seg-RS
"$PYTHON" tools/prepare_ovs44.py --root "$DATA_ROOT" --output data/ovs44
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$SAM_ROOT:/home/Think2Seg-RS/src/open-r1-multimodal/src" \
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/open-r1-multimodal/src/open_r1/grpo_geo_ultra.py \
  --dataset_name none \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --max_prompt_length 512 \
  --max_completion_length 256 \
  --num_generations 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_steps 1 \
  --logging_steps 1 \
  --learning_rate 1e-5 \
  --bf16 \
  --torch_dtype bfloat16 \
  --report_to none \
  --gradient_checkpointing true \
  --attn_implementation sdpa \
  --sam_model_size small \
  --sam_root "$SAM_ROOT" \
  --use_datasets ovs44 \
  --ovs44_manifest /home/Think2Seg-RS/data/ovs44/train.json \
  --earthreason_resize_size 280 \
  --freeze_vision_modules true \
  --beta 0 \
  --use_peft true \
  --lora_r 8 \
  --lora_alpha 16 \
  --save_strategy no
