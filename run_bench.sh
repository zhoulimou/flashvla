#!/bin/bash
# Thin wrapper around sim_eval/libero/bench_npu.py for the numbers quoted in
# docs/ascend_baseline.md. Kept in the repo so the Reproduce section works on a
# fresh clone (previously an untracked /tmp script).
#
#   CARD=5 BENCH_CHUNKS=40 BENCH_WARMUP=4 BENCH_STEPS=60 \
#   BENCH_OVERLAP=0 BENCH_VANILLA=1 BENCH_TAG=bench_final_c5 bash run_bench.sh
#
# bench_npu.py reads its own switches from the environment (BENCH_*), so this
# wrapper only has to select the chip, the runtime env and the log location.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR/sim_eval/libero" || exit 1

export CARD="${CARD:-0}"
export BENCH_CHUNKS="${BENCH_CHUNKS:-50}"
export BENCH_WARMUP="${BENCH_WARMUP:-5}"
export BENCH_STEPS="${BENCH_STEPS:-120}"
export BENCH_OVERLAP="${BENCH_OVERLAP:-0,1,2,3}"
export BENCH_VANILLA="${BENCH_VANILLA:-1}"
export BENCH_VANILLA_ITERS="${BENCH_VANILLA_ITERS:-5}"
export BENCH_TAG="${BENCH_TAG:-bench}"
LOG_DIR="${LOG_DIR:-/home/data/flashvla/logs}"

export ASCEND_RT_VISIBLE_DEVICES="$CARD"
POLICY_PATH="${POLICY_PATH:-/home/data/flashvla/weights/flashvla-pi05-libero}"

# Ascend runtime: the toolkit's set_env.sh dereferences LD_LIBRARY_PATH before
# defining it, which trips `set -u`.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

# Never block on a huggingface.co HEAD request for a model that is already local.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# MuJoCo/EGL: the README's NVIDIA EGL vendor json does not exist on Ascend.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${BENCH_TAG}_c${CARD}.log"
echo "[run_bench] card=$CARD chunks=$BENCH_CHUNKS warmup=$BENCH_WARMUP steps=$BENCH_STEPS"
echo "[run_bench] overlap=$BENCH_OVERLAP vanilla=$BENCH_VANILLA tag=$BENCH_TAG log=$LOG"

python -u bench_npu.py \
  --policy.path="$POLICY_PATH" \
  --policy.device=npu \
  --policy.compile_model=false \
  --policy.fuse_qkv=false \
  --policy.fuse_gate_up=false \
  2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
