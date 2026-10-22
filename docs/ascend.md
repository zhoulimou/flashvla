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

## Measured (single 910B3, `bench_npu.py`)

Full baseline report: [`docs/ascend_baseline.md`](ascend_baseline.md). Native code path
(appliance-level device plumbing only), `n_action_steps=10`, batch 1, sync decoding, 40 chunks
per run on three different chips:

| Metric | Value |
|---|---|
| FlashVLA chunk latency | **124.3 / 134.4 / 129.8 ms** (median 123.9 / 134.2 / 129.1) |
| FlashVLA **Time/Step** | **12.43 / 13.44 / 12.98 ms** |
| stage breakdown | encode ~35-38 + prefill ~40-45 + denoise ~47-52 ms -> **56-62 % is observation-prefix work** |
| vanilla pi0.5, prefix cached once per chunk | 559 / 599 / 550 ms per chunk -> **55.9 / 59.9 / 55.0 ms per step** |
| vanilla pi0.5, prefix recomputed per denoise step | 1246 / 1323 / 1238 ms -> 124.6 / 132.3 / 123.8 ms per step |
| **same-device speedup (honest)** | **4.50 / 4.45 / 4.24x** (pessimistic variant: 10.02 / 9.85 / 9.54x) |
| peak HBM | 8424 MB |

Official CUDA numbers for reference (README, RTX 4090, 2 views, `benchmarks/benchmark_latency.py`):
pi0.5 45.8 ms/chunk, FlashVLA 26.7 ms/chunk -> 1.72x. So the FlashVLA algorithmic win is
**larger on Ascend (4.2-4.5x) than on a 4090 (1.72x)**, because one denoise step is relatively
more expensive here (no fused kernels), while the absolute latency is ~4.9x the 4090.

## LIBERO evaluation

Run with `--policy.device=npu --policy.n_action_steps=10 --env.max_parallel_tasks=1
--eval.batch_size=1 --inference_overlap_steps=0 --seed=1000`.

10 episodes per task, seed 1000, sync decoding (`--inference_overlap_steps=0`):

| Suite | pc_success | avg steps/ep | s/episode | ms/action step | eval wall time |
|---|---|---|---|---|---|
| libero_spatial | **96.0 %** (96/100) | 132.2 | 82.4 | 622.9 | 2 h 35 m |
| libero_object | **99.0 %** (99/100) | 160.3 | 56.3 | 351.1 | 1 h 46 m |
| libero_goal | **99.0 %** (99/100) | 133.3 | 68.7 | 515.4 | 2 h 10 m |
| libero_10 | **96.0 %** (96/100) | 274.7 | 102.0 | 371.5 | 3 h 03 m |

Wall time per environment step is dominated by MuJoCo offscreen rendering on CPU
(llvmpipe) and observation preprocessing — the policy accounts for only ~13 ms of the
~600 ms per step. That is why the LIBERO numbers here cannot be compared to the paper's
`Time/Step` metric directly: the paper measures policy time, this table measures whole-loop
wall time. Paper reference: π0.5 96.9 average, FlashVLA 97.9–98.3 average on LIBERO.
Our four-suite average (10 episodes per task): **97.5**.

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

   **But at the shipped LIBERO config it cannot be set.** `AsyncStreamingActionManager`
   enforces `overlap_steps + n_action_steps <= chunk_size` when RTC is on, and the
   checkpoint ships `chunk_size = 10` with `n_action_steps = 10`, so any
   `overlap_steps >= 1` raises `ValueError: skip_stale_actions requires overlap_steps +
   n_action_steps <= chunk_size`. There are only two consistent options:

   * `n_action_steps=10`, `SKIP_STALE_ACTIONS=0` — RTC off, overlap >= 3 degrades
     accuracy (the 2/5 above);
   * `n_action_steps=5`, `SKIP_STALE_ACTIONS=1` — the constraint is satisfied
     (`5 + 5 <= 10`) and RTC can realign, at the cost of a different execution
     cadence than the reference numbers.

   Any overlap sweep measured with RTC off is not comparable to an RTC-enabled run;
   `bench_npu.py` records the setting as `skip_stale_actions` in its JSON output.

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
* **Never launch the same eval twice.** A second `eval.py` with the same `--output_dir` does
  not replace the first one: both keep running, they share `libero10.log` and
  `eval_info.partial.json`, and the interleaved progress bars make the run look like it is
  restarting. `pgrep -af "sim_eval/libero/eval.py"` before launching, and kill duplicates.

Notes / non-goals:

* `config.json` sets `"dtype": "bfloat16"` and `__init__` calls
  `to_bfloat16_for_selected_params()`, so the backbone already runs bf16 (the fp32 params
  are `patch_embedding`/`position_embedding`/layernorms by design). Do not wrap inference in
  `torch.autocast("npu")` — it was measured **1.9× slower** (256.0 vs 134.2 ms/chunk) because
  every op re-casts fp32 inputs.

## Fast policy construction (`flashvla/startup.py`)

`PreTrainedPolicy.from_pretrained` builds the policy from its config — randomly initialising
all 3B weights on CPU — and only then loads the safetensors checkpoint. That discarded
initialisation dominated start-up. `flashvla.startup.no_random_init()` wraps transformers'
`no_init_weights()` (which patches `torch.nn.init.*` and `PreTrainedModel.init_weights` to
no-ops) around `make_policy`:

| path | build time |
|---|---|
| default (`from_pretrained` as-is) | **320.5 s** |
| `with no_random_init():` | **2.8 s** (114.7× faster) |

Verified with `sim_eval/libero/bench_startup.py` on the same weights: all 813 parameters are
present in the checkpoint, and 8 spot-checked tensors match the safetensors bit-for-bit
(`max_abs_diff = 0.0`). Peak HBM is unchanged (8308 MB).

```bash
ASCEND_RT_VISIBLE_DEVICES=6 python sim_eval/libero/bench_startup.py \
    --policy.path=/home/data/flashvla/weights/flashvla-pi05-libero \
    --policy.device=npu --env.type=libero \
    --output_dir=/home/data/flashvla/logs/startup_out
```

`sim_eval/libero/eval.py` already wraps its `make_policy` call and logs `Policy built in Xs`.
`bench_npu.py` does not need it (its construction cost is outside the timed regions).

## Reproduce the numbers

```bash
CARD=5 BENCH_CHUNKS=40 BENCH_STEPS=120 BENCH_OVERLAP=0,1,2,3 bash run_bench.sh
```
