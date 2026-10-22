#!/bin/bash
# Single-suite LIBERO eval on one Ascend chip, for the numbers in
# docs/ascend_baseline.md. The upstream eval.sh targets CUDA (policy.device=cuda,
# CUDA_VISIBLE_DEVICES); this wrapper is the NPU equivalent and used to live as
# an untracked /tmp script, which made the Reproduce section unreproducible.
#
#   SUITE=libero_spatial EPS=10 CARD=3 TAG=spatial10 bash run_eval_generic.sh
#
# Env: SUITE (one suite), EPS (episodes per task), CARD (chip id), TAG (output
# dir stem), SEEDS, OMP_THREADS, N_ACTION_STEPS, OVERLAP_STEPS, POLICY_PATH.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR" || exit 1

SUITE="${SUITE:-libero_spatial}"
EPS="${EPS:-10}"
CARD="${CARD:-0}"
TAG="${TAG:-${SUITE}_npueval}"
SEEDS="${SEEDS:-1000}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
OVERLAP_STEPS="${OVERLAP_STEPS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/data/flashvla/logs}"
POLICY_PATH="${POLICY_PATH:-/home/data/flashvla/weights/flashvla-pi05-libero}"

export ASCEND_RT_VISIBLE_DEVICES="$CARD"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

# Small-tensor CPU preprocessing pays an OpenMP sync tax on many-core hosts;
# pinning the thread count reproduces the low-core latency quoted in the doc.
export OMP_NUM_THREADS="${OMP_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

OUT="$OUTPUT_ROOT/${TAG}_out"
LOG="$OUTPUT_ROOT/${TAG}.log"
mkdir -p "$OUT"

echo "[run_eval] suite=$SUITE chip=$CARD eps=$EPS seeds=$SEEDS tag=$TAG"
echo "[run_eval] out=$OUT log=$LOG"
echo "[run_eval] note: at most 3 concurrent evals — EGL contexts are limited"

python -u sim_eval/libero/eval.py \
  --policy.path="$POLICY_PATH" \
  --policy.device=npu \
  --policy.compile_model=false \
  --policy.fuse_qkv=false \
  --policy.fuse_gate_up=false \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPS" \
  --inference_overlap_steps="$OVERLAP_STEPS" \
  --seed="$SEEDS" \
  --output_dir="$OUT" \
  2>&1 | tee "$LOG"

rc="${PIPESTATUS[0]}"
if [ -f "$OUT/eval_info.json" ]; then
  python3 - "$OUT/eval_info.json" "$CARD" "$SUITE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["overall"]
print(f"  [chip={sys.argv[2]}] suite={sys.argv[3]}  pc_success={d.get('pc_success'):.1f}")
PY
fi
exit "$rc"
