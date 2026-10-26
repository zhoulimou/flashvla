#!/bin/bash
# Run one official latency benchmark (benchmarks/benchmark_latency.py) on an
# Ascend NPU chip. Everything else is the upstream script, unmodified.
#
# Usage:  bash run_latency_npu.sh <config.yaml> [chip] [log_tag]
# e.g.    ASCEND_RT_VISIBLE_DEVICES=4 bash run_latency_npu.sh npu_latency_base.yaml 4 L-base
set -u
CFG="${1:?usage: run_latency_npu.sh <config.yaml> [chip] [tag]}"
CARD="${2:-4}"
TAG="${3:-$(basename "$CFG" .yaml)}"

REPO=/home/z00633863/flashvla
OUTROOT=${OUTROOT:-/home/data/flashvla/logs/paperL}
mkdir -p "$OUTROOT"
LOG="$OUTROOT/${TAG}.log"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$CARD}"
export PATH=/root/miniconda3/envs/flashvla-npu/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_THREADS:-16}
export MKL_NUM_THREADS=${OMP_THREADS:-16}
export PYTHONUNBUFFERED=1
export FLASHVLA_FAST_STARTUP=1
export MUJOCO_GL=egl
unset EGL_PLATFORM LIBGL_ALWAYS_SOFTWARE

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd "$REPO" || exit 1
echo "[$(date '+%F %T')] START $TAG chip=$ASCEND_RT_VISIBLE_DEVICES cfg=$CFG" | tee "$LOG"

/root/miniconda3/envs/flashvla-npu/bin/python benchmarks/benchmark_latency.py \
  --config_path="$CFG" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] END $TAG rc=$rc" | tee -a "$LOG"
exit "$rc"
