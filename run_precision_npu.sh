#!/bin/bash
# Run one paper-reproduction precision group with the *upstream* eval.sh.
#
# Usage:
#   bash run_precision_npu.sh <tag> <policy_path> <n_action_steps> <overlap> \
#                             <n_episodes> <seed> [compile] [fuse]
#
# Example (FlashVLA d=0, 2000 episodes, chips 4-7):
#   GPUS="0 1 2 3" bash run_precision_npu.sh flashvla_d0 \
#       /home/data/flashvla/weights/flashvla-pi05-libero 10 0 50 1000 false false
set -u
TAG="${1:?usage: run_precision_npu.sh <tag> <policy_path> <n_action_steps> <overlap> <n_episodes> <seed> [compile] [fuse]}"
POLICY_PATH="${2:?}"
N_ACTION_STEPS="${3:?}"
OVERLAP="${4:?}"
N_EPISODES="${5:?}"
SEED="${6:?}"
COMPILE_MODEL="${7:-false}"
FUSE="${8:-false}"

REPO=/home/z00633863/flashvla
export GPUS="${GPUS:-0 1 2 3}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/data/flashvla/logs/paperE}"
export SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
export OMP_THREADS="${OMP_THREADS:-8}"

export PATH=/root/miniconda3/envs/flashvla-npu/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export FLASHVLA_FAST_STARTUP=1
export MUJOCO_GL=egl
unset EGL_PLATFORM LIBGL_ALWAYS_SOFTWARE

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

cd "$REPO" || exit 1
mkdir -p "$OUTPUT_ROOT"

echo "[$(date '+%F %T')] START $TAG | policy=$POLICY_PATH | N=$N_ACTION_STEPS overlap=$OVERLAP eps=$N_EPISODES seed=$SEED compile=$COMPILE_MODEL fuse=$FUSE | chips=$GPUS"

POLICY_DEVICE=npu \
POLICY_PATH="$POLICY_PATH" \
N_ACTION_STEPS="$N_ACTION_STEPS" \
COMPILE_MODEL="$COMPILE_MODEL" \
FUSE="$FUSE" \
N_EPISODES="$N_EPISODES" \
bash sim_eval/libero/eval.sh "$OVERLAP" "$SEED"

echo "[$(date '+%F %T')] END $TAG rc=$?"
