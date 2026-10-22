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

# Honour an explicitly exported ASCEND_RT_VISIBLE_DEVICES (useful for ad-hoc runs);
# CARD is only a fallback so the documented CARD=N form keeps working.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$CARD}"

# The flashvla-npu conda env is not on PATH in a non-interactive ssh shell;
# the system python3 has no torch_npu/numpy. Pin the interpreter explicitly.
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/flashvla-npu/bin/python}"
export PATH="/root/miniconda3/envs/flashvla-npu/bin:$PATH"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Only MUJOCO_GL is needed. EGL_PLATFORM / LIBGL_ALWAYS_SOFTWARE must stay
# unset -- setting either one segfaults robosuite's offscreen EGL context in
# make_env on this box (verified against the 2026-10-22 known-good run).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset EGL_PLATFORM LIBGL_ALWAYS_SOFTWARE

# Small-tensor CPU preprocessing pays an OpenMP sync tax on many-core hosts;
# pinning the thread count reproduces the low-core latency quoted in the doc.
export OMP_NUM_THREADS="${OMP_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

OUT="$OUTPUT_ROOT/${TAG}_out"
LOG="$OUTPUT_ROOT/${TAG}.log"
mkdir -p "$OUT"

echo "[run_eval] suite=$SUITE chip=$CARD eps=$EPS seeds=$SEEDS tag=$TAG"
echo "[run_eval] out=$OUT log=$LOG"
# Measured on this box: 7 concurrent eval processes each holding one MuJoCo
# EGL context run fine, and a single process was also verified with 5 contexts;
# 7 procs x 5 envs (35 contexts) all built successfully. The earlier "at most 3
# concurrent" note was a guess, not a limit of the software rasteriser.

"$PYTHON_BIN" -u sim_eval/libero/eval.py \
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
  "$PYTHON_BIN" - "$OUT/eval_info.json" "$CARD" "$SUITE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["overall"]
print(f"  [chip={sys.argv[2]}] suite={sys.argv[3]}  pc_success={d.get('pc_success'):.1f}")
PY
fi
exit "$rc"
