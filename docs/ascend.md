# Running π0.5 + FlashVLA on Ascend NPU (910B3 / CANN)

Status: **inference-only** port, validated on 8× Ascend 910B3 (openEuler 22.03, CANN 8.x,
torch 2.9.0+cpu + torch_npu 2.9.0, Python 3.12). No training support.

## What was changed

| File | Change |
|---|---|
| `flashvla/npu_support.py` (new) | `apply()` monkey-patches `lerobot.utils.device_utils`: accepts `npu` in `is_torch_device_available`, prefers NPU in `auto_select_torch_device`, reports AMP available. Mirrored into `lerobot.configs.policies` / `lerobot.policies.factory` (by-value imported symbols). |
| `sim_eval/libero/eval.py` | `get_safe_torch_device()` returns `torch.device("npu:0")` for `--policy.device=npu`; imports `flashvla.npu_support` before LeRobot builds the policy. |
| `sim_eval/libero/bench_npu.py` (new) | Latency benchmark: policy-only Time/Step, encode/prefill/denoise breakdown, vanilla 10-step decode baseline, async-overlap manager steps. |
| `docs/ascend.md` | This file. |

Why the shim is needed: LeRobot's `is_torch_device_available()` whitelists only
`cuda/mps/xpu/cpu`, and `PI05FlashVLAConfig.__post_init__` calls it during draccus parsing,
so `--policy.device=npu` fails before any model code runs.

## Environment notes

* `source /usr/local/Ascend/ascend-toolkit/set_env.sh` **before** python, otherwise
  `import torch` (auto-loading torch_npu) fails with
  `ImportError: libhccl.so: cannot open shared object file`.
* `pip install decorator sympy attrs cloudpickle absl-py psutil grpcio protobuf` — CANN op
  compilation (`GEInitialize`, TeFusion) imports `decorator` from its embedded Python and
  dies otherwise (`Environment_Error_Import_Python_Module_Failed(EC0010)`).
* Tokenizer: the checkpoint points at `google/paligemma-3b-pt-224` (gated). Without an HF
  token, seed a local cache entry from an ungated Gemma-2B tokenizer (`unsloth/gemma-2b`):
  it matches PaliGemma for normal text ids (vocab 256000, pad 0, eos 1, bos 2); only the
  1152 extra `<loc*>`/`<seg*>` ids differ, which LIBERO language instructions never use.
  `refs/main` must be written **without a trailing newline**, otherwise
  `huggingface_hub.try_to_load_from_cache()` never matches.

## Run

```bash
export ASCEND_RT_VISIBLE_DEVICES=4 MUJOCO_GL=egl OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
python sim_eval/libero/eval.py \
  --policy.path=/home/data/flashvla/weights/flashvla-pi05-libero \
  --policy.device=npu --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial \
  --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=10 \
  --inference_overlap_steps=0 --seed=1000 \
  --output_dir=/home/data/flashvla/logs/eval_spatial_out
```

## Measured (single 910B3, checkpoint dtype bf16, n_action_steps=10, `bench_npu.py`)

| Metric | Value |
|---|---|
| FlashVLA chunk latency | **131.5 ms** (p50 131.3, p95 135.9), 40 chunks |
| FlashVLA **Time/Step** | **13.15 ms** |
| stage breakdown | encode 39.0 + prefill 44.1 + denoise 52.2 ms → **61% is observation-prefix recompute** |
| vanilla π0.5 10-step decode | 1271.4 ms/chunk → 127.1 ms/step |
| **same-device speedup** | **9.67×** (paper reports 2.62× on GPU) |
| async manager step (no env) | 13.00 / 12.70 / 12.67 / 12.46 ms for overlap 0/1/2/3 |
| peak HBM | 8424 MB |
| LIBERO eval, spatial 10 tasks × 1 ep | 10/10 success, 630.7 ms/step wall-clock (MuJoCo env dominates) |

Why the speedup is larger than on GPU: FlashVLA amortizes the observation-prefix forward
over `n_action_steps`; on Ascend that prefix forward has no fused kernel equivalent, so
vanilla decoding pays it 10× per chunk. Same policy, same hardware → the algorithmic gain is
larger, which is the interesting NPU-side result.

## LIBERO evaluation

Run with `--policy.device=npu --policy.n_action_steps=10 --env.max_parallel_tasks=1
--eval.batch_size=1 --inference_overlap_steps=0 --seed=1000`.

10 episodes per task, seed 1000, sync decoding (`--inference_overlap_steps=0`):

| Suite | pc_success | avg steps/ep | s/episode | ms/action step | eval wall time |
|---|---|---|---|---|---|
| libero_spatial | **96.0 %** (96/100) | 132.2 | 82.4 | 622.9 | 2 h 35 m |
| libero_object | **99.0 %** (99/100) | 160.3 | 56.3 | 351.1 | 1 h 46 m |
| libero_goal | **99.0 %** (99/100) | 133.3 | 68.7 | 515.4 | 2 h 10 m |
| libero_10 | running (10 tasks × 10 ep) | – | ~120 | ~600 | ~3.5 h expected |

Wall time per environment step is dominated by MuJoCo offscreen rendering on CPU
(llvmpipe) and observation preprocessing — the policy accounts for only ~13 ms of the
~600 ms per step. That is why the LIBERO numbers here cannot be compared to the paper's
`Time/Step` metric directly: the paper measures policy time, this table measures whole-loop
wall time. Paper reference: π0.5 96.9 average, FlashVLA 97.9–98.3 average on LIBERO.

### Async chunk overlap (`--inference_overlap_steps`, `sweep_overlap.py`)

Task `libero_spatial/0`, 5 episodes per setting, same seed, policy loaded once:

| overlap_steps | pc_success | s/episode | ms/action step |
|---|---|---|---|
| 0 (sync) | 100 % | 66.8 | 643.2 |
| 1 | 100 % | 69.4 | 633.0 |
| 2 | 100 % | 72.2 | 647.9 |
| 3 (no RTC realignment) | **40 %** | 123.2 | 573.0 |

Two findings:

1. **Overlap buys nothing on this box.** Policy time is ~13 ms/action step while the
   environment loop costs ~600 ms/action step, so there is at most a ~2 % headroom to hide;
   the manager's own bookkeeping eats it. Upstream's async mode assumes CUDA-graph compiled
   dispatch (`eval.sh` forces `COMPILE_MODEL=true` when `OVERLAP_STEPS>0`).
2. **`torch.compile` does not work in this NPU env** — `torch.compile` raises
   `BackendCompilerFailed backend='inductor' ... ModuleNotFoundError: No module named 'triton'`,
   and the Inductor backend has no Ascend target here. So the graph-compiled async path
   cannot be reproduced as-is; all numbers above use sync decoding.
3. Overlap without `skip_stale_actions` (RTC realignment) reuses actions computed
   `overlap_steps` observations ago; at overlap=3 success collapses (2/5). If async is
   enabled, `SKIP_STALE_ACTIONS=1` must be set.

Partial results are written after every completed task to
`<output_dir>/eval_info.partial.json` (atomically) and `eval_info.partial.jsonl`, so a
crashed run keeps the tasks it already finished. The final `eval_info.json` still contains
`per_task`, `per_group`, `overall` and now also a `progress` block.

Operational pitfalls (all verified on 192.168.45.9):

* **Do not kill "duplicate" eval processes.** Each eval has 3 extra children with the same
  cmdline; they are the CANN TBE kernel-cache `SyncManager` daemons
  (`tbe/common/repository_manager/utils/multiprocess_util.py:48`). They inherit the cmdline
  and log fd but never write logs. Killing them breaks op-compile cache lookups.
* **EGL is scarce.** Each eval process creates 10 MuJoCo offscreen contexts; running 4 evals
  (40 contexts) makes a new one die instantly with
  `OpenGL.raw.EGL._errors.EGLError: <exception str() failed>`. Keep ≤3 concurrent evals, or
  launch them staggered.
* **`HF_HUB_OFFLINE=1` + LIBERO assets.** `libero/libero/__init__.py:get_assets_path()`
  falls back to a *nonexistent* path when the HF download raises, and MuJoCo may then
  segfault. We keep `HF_HUB_OFFLINE=1` (the tokenizer cache alias needs it) and make the
  lookup deterministic with a symlink:
  `ln -s /root/.cache/libero/assets <site-packages>/libero/libero/assets`.
* `AsyncStreamingActionManager` is constructed per episode; that is cheap and only matters
  if you enable `--policy.compile_model=true` (graph capture per episode would be costly).

Notes / non-goals:

* `config.json` sets `"dtype": "bfloat16"` and `__init__` calls
  `to_bfloat16_for_selected_params()`, so the backbone already runs bf16 (the fp32 params
  are `patch_embedding`/`position_embedding`/layernorms by design). Do not wrap inference in
  `torch.autocast("npu")` — it was measured **1.9× slower** (256.0 vs 134.2 ms/chunk) because
  every op re-casts fp32 inputs.
* Load time is ~7 min: LeRobot builds fp32 params on CPU (random init) before loading the
  safetensors; not addressed yet.

## Reproduce the numbers

```bash
CARD=5 BENCH_CHUNKS=40 BENCH_STEPS=120 BENCH_OVERLAP=0,1,2,3 bash run_bench.sh
```
