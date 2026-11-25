#!/bin/bash
# .8: launch FlashVLA fine-tuning on the LIBERO-Pro LeRobot dataset.
# Usage: bash train_pro_ft.sh <config.yaml> [num_cards]
set -u
CONFIG=${1:?usage: bash train_pro_ft.sh <config.yaml> [num_cards]}
NUM=${2:-8}
REPO=/home/z00633863/flashvla
cd "$REPO" || exit 1

export PATH="/root/miniconda3/envs/flashvla-npu/bin:$PATH"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/data/hf_home
export HF_DATASETS_CACHE=/home/data/flashvla/hf_datasets_cache
export TMPDIR=/home/data/tmp
mkdir -p /home/data/tmp
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

accelerate launch --multi_gpu --num_processes="$NUM" \
  --module flashvla.train --config_path="$CONFIG"
