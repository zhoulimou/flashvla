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

* **MuJoCo/EGL: leave `EGL_PLATFORM` and `LIBGL_ALWAYS_SOFTWARE` unset.** Only
  `MUJOCO_GL=egl` is required. Setting `EGL_PLATFORM=surfaceless` or
  `LIBGL_ALWAYS_SOFTWARE=1` makes the second and later `make_env()` call either segfault
  (`exit 139`, no Python traceback) or abort with
  `OpenGL.raw.EGL._errors.EGLError: <exception str() failed>` — the first env often builds
  fine, so the failure looks like a model problem. Both wrappers now `unset` them
  explicitly. `LIBGL_ALWAYS_SOFTWARE=1` (LLVMpipe) was a leftover from the initial
  `Cannot initialize a EGL device display` fix; installing `mesa-dri-drivers` for swrast
  was the actual fix, and forcing software rendering on top of it is what broke.
* `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: pin to 8 for benchmarks. On a 256-core host the
  small batch-1 CPU preprocessing pays an OpenMP sync tax that shows up as run-to-run
  noise; 8 matches the low-core behaviour the latency doc quotes.

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

Two budgets, both seed 1000 sync decoding (`--inference_overlap_steps=0`). The 500-episode
run is the number to quote:

| Suite | 50 ep/task (500 ep) | 10 ep/task (100 ep) | avg steps/ep | s/episode | ms/action step |
|---|---|---|---|---|---|
| libero_spatial | **99.0 %** | 96.0 % | 128.3 | 78.6 | 612.6 |
| libero_object | **99.4 %** | 99.0 % | 158.2 | 55.3 | 349.9 |
| libero_goal | **97.4 %** | 99.0 % | 135.5 | 67.9 | 501.2 |
| libero_10 | **96.0 %** | 96.0 % | 276.4 | 107.4 | 388.5 |
| **average** | **98.0** | 97.5 | | | |

Paper reference: π0.5 96.9 average, FlashVLA 97.9 (`d=0`) – 98.3 (`d=2`) average on LIBERO,
2000 episodes × 5 seeds. Our 500-episode average of 98.0 sits inside that band, so the port
shows no measurable accuracy loss. At 500 episodes the standard error is ±0.9 %, so
differences under ~1.8 points are not resolvable from a single seed.

Wall time per environment step is dominated by MuJoCo offscreen rendering on CPU
(llvmpipe) and observation preprocessing — measured decomposition puts the policy at
2-3 % of an end-to-end step (`sim_eval/libero/eval.py` rollout loop instrumented with
`perf_counter`). That is why the LIBERO numbers here cannot be compared to the paper's
`Time/Step` metric directly: the paper measures policy time, this table measures whole-loop
wall time. See [docs/ascend_baseline.md](ascend_baseline.md) for both, spelled out.

Per-task successes out of 50 (spatial / object / goal / libero_10):
`[50,50,50,50,48,50,50,49,48,50]` / `[50,50,50,47,50,50,50,50,50,50]` /
`[50,49,49,40,50,49,50,50,50,50]` / `[50,49,50,45,47,50,49,50,41,49]`.

### Async chunk overlap (`--inference_overlap_steps`)

Two different things are called "overlap" in this repo and they must not be compared
across the two tables below.

#### Accuracy: whole-loop rollouts (`sweep_overlap.py`)

Task `libero_spatial/0`, 5 episodes per setting, same seed, policy loaded once. The
manager is built with `skip_stale_actions=False`, so no RTC realignment is in play:

| overlap_steps | pc_success | s/episode | ms/action step |
|---|---|---|---|
| 0 (sync) | 100 % | 66.8 | 643.2 |
| 1 | 100 % | 69.4 | 633.0 |
| 2 | 100 % | 72.2 | 647.9 |
| 3 (no RTC realignment) | **40 %** | 123.2 | 573.0 |

Without `skip_stale_actions` the manager replays actions computed `overlap_steps`
observations earlier, and at overlap=3 success collapses to 2/5. This is why the
evaluation table above uses sync decoding.

#### Latency: policy-only, simulated env step (`bench_npu.py`)

`measure_manager()` builds `AsyncStreamingActionManager(..., skip_stale_actions=False)`
and charges a synthetic per-step env cost `BENCH_ENV_MS`. 40 chunks, 120 timed steps,
4 chips in parallel, `ENV_MS=0` vs `ENV_MS=600`:

| ENV_MS | overlap 0 | overlap 1 | overlap 2 | overlap 3 |
|---|---|---|---|---|
| 0    | 12.72 ms | 12.95 ms | 12.96 ms | 12.95 ms |
| 0    | 12.02 ms | 11.96 ms | 12.10 ms | 12.12 ms |
| 600  | 613.96 ms | 613.06 ms | 612.58 ms | 612.49 ms |
| 600  | 612.04 ms | 612.09 ms | 612.13 ms | 612.11 ms |

At `ENV_MS=600` the stall is only 12.1–13.1 ms of a ~612 ms step, i.e. the policy is
2 % of the loop and overlap can hide at most that much: measured gain from overlap 0→3 is
1.5 ms (0.24 %), within run-to-run noise. At `ENV_MS=0` there is nothing to overlap with,
so overlap is flat-to-slightly-negative (12.72 → 12.95 ms).

Three findings:

1. **Overlap buys nothing on this box.** The environment loop costs ~600 ms/action step
   while the policy costs ~13 ms, so there is at most ~2 % of headroom to hide and the
   manager's own bookkeeping eats it. Upstream's async mode also assumes CUDA-graph
   compiled dispatch (`eval.sh` forces `COMPILE_MODEL=true` when `OVERLAP_STEPS>0`).
2. **`torch.compile` does not work in this NPU env** — `torch.compile` raises
   `BackendCompilerFailed backend='inductor' ... ModuleNotFoundError: No module named 'triton'`,
   and the Inductor backend has no Ascend target here (`torchair` is not installed with
   CANN 8.5.0). So the graph-compiled async path cannot be reproduced as-is; every number
   above uses eager execution with `compile_model=false`.
3. **`SKIP_STALE_ACTIONS=1` is not usable at the current shape.** `async_manager.py`
   rejects `skip_stale_actions` unless `overlap_steps + n_action_steps <= chunk_size`.
   With this checkpoint (`num_buffer_slots=4`, `chunk_size=10`, `n_action_steps=10`)
   that reduces to `overlap_steps <= 0`, so any overlap ≥ 1 raises
   `ValueError: skip_stale_actions requires overlap_steps + n_action_steps <= chunk_size`.
   Enabling RTC realignment therefore also requires lowering `n_action_steps`
   (e.g. 5) — a configuration change that has not been evaluated for accuracy here.
   Do not assume `SKIP_STALE_ACTIONS=1` is a drop-in switch.

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
