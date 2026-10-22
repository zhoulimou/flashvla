# Ascend NPU baseline — pi0.5 + FlashVLA inference

Reference numbers for the unoptimised, native code path. Use this as the starting point
for any Ascend optimisation work: every number below was measured **without** kernel
fusion, `torch.compile`, autocast, batching, prefix reuse across chunks, or async overlap.

Measured 2026-10-22 on a shared Ascend 910B3 node (see [docs/ascend.md](ascend.md) for
how to run things).

## Scope

- code: FlashVLA at the `feat/ascend-npu` branch, i.e. upstream `4e0b08f` plus device
  plumbing only (`flashvla/npu_support.py`, an `npu` branch in `get_safe_torch_device`,
  the benchmark/eval tooling). No model or sampler change.
- policy: `flashvla-pi05-libero` checkpoint as shipped (safetensors weights are bf16,
  `config.dtype = "bfloat16"`; the few norm/embedding tensors kept in fp32 by the model
  stay fp32 on purpose).
- decoding: `--policy.n_action_steps=10`, `num_buffer_slots=4`, `chunk_size=10`,
  batch 1, 2 camera views, `compile_model=false`, `fuse_qkv=false`, `fuse_gate_up=false`,
  synchronous decoding (`--inference_overlap_steps=0`).
- `FLASHVLA_FAST_STARTUP=1` is used by the benchmark harness only (it skips the discarded
  random weight init: 320 s -> 2.8 s, weights verified bit-identical). It does not touch
  any measured region.

## Hardware / software

| item | value |
|---|---|
| accelerator | Ascend 910B3, 1 chip of 8 (64 GB HBM), shared with other users |
| host | openEuler 22.03 LTS, 256 cores, 2 TB RAM; MuJoCo renders offscreen on CPU (llvmpipe) |
| runtime | python 3.12, torch 2.9.0+cpu, torch_npu 2.9.0, lerobot 0.5.1 |
| weights | `/home/data/flashvla/weights/flashvla-pi05-libero` |

## Latency — policy forward only

`bench_npu.py`, 4 warm-up chunks then 40 timed chunks, three different chips:

| chip | run | chunk mean | p50 | p95 | ms / action step | encode | prefill | denoise |
|---|---|---|---|---|---|---|---|---|
| 4 | A1 | 127.3 ms | 127.4 | 130.6 | **12.73** | 36.0 | 41.5 | 48.6 |
| 7 | A2 | 122.3 ms | 122.0 | 124.6 | **12.23** | 35.1 | 39.8 | 46.5 |
| 5 | A3 | 128.5 ms | 127.1 | 136.4 | **12.85** | 37.1 | 41.7 | 48.6 |
| 6 | A4 | 123.8 ms | 123.0 | 128.7 | **12.38** | 34.9 | 40.4 | 47.4 |
| 5 | earlier | 124.3 ms | 123.9 | 132.7 | 12.43 | 34.7 | 40.3 | 46.7 |
| 6 | earlier | 134.4 ms | 134.2 | 135.5 | 13.44 | 38.5 | 44.7 | 52.1 |
| 7 | earlier | 129.8 ms | 129.1 | 137.9 | 12.98 | 35.8 | 41.7 | 49.3 |

- Working range across all 7 runs on 4 chips: **122-134 ms per chunk, 12.2-13.4 ms per
  action step**. The node is shared, so treat a ±5 % spread as noise.
- Rows `A1-A4` are from a single sweep that ran four chips concurrently (see
  [Concurrency](#concurrency-four-chips-at-once)); `earlier` rows were measured one chip
  at a time. The two groups agree, so running four chips in parallel does not perturb the
  per-chip latency by more than the noise floor.
- Observation-prefix work (`encode` + `prefill`) is **56-62 %** of the chunk. That is the
  single largest Ascend-side optimisation target: reusing the prefix across chunks (or a
  cheaper image encoder path) would remove most of it.
- Peak HBM: **8424 MB** per process at batch 1.

## Vanilla pi0.5 decoding, same weights, same chip

The honest question is "what would the original sampler cost here".

The baseline is **unmodified pi0.5**: the stock flow-matching sampler that draws one
action chunk with `denoise_steps = 10` full forwards. `bench_npu.py` measures it two
ways, and the difference between them is large enough that the choice must be stated
explicitly in any comparison:

| variant | what it models | chunk latency | ms / action step | FlashVLA speedup |
|---|---|---|---|---|
| **(a) prefix cached once per chunk** | what the real flow-matching sampler does: encode + prefill once, then 10 cheap denoise forwards reusing the KV cache | 559 / 599 / 550 ms | **55.0-59.9** | **4.24-4.50x** |
| **(b) prefix recomputed every denoise step** | encode + prefill repeated inside all 10 denoise steps; no cross-step KV cache | 1238 / 1246 / 1323 ms | 123.8-132.3 | 9.54-10.02x |

Cross-check from the concurrent four-chip sweep (full numbers in the table below):
cached **54.4-58.7 ms/step** at **4.43-4.61x**; recomputed **120.8-127.5 ms/step** at
**9.80-10.02x**.

**Headline: 4.2-4.6x, using variant (a).** Variant (b) is the pessimistic figure an
earlier version of this harness reported (9.5-10x); it overstates the gain because no real
pi0.5 sampler recomputes the observation prefix on every denoise step. The two numbers
must not be mixed: (b) is a useful upper bound on the theoretical headroom of prefix
reuse, not a like-for-like comparison. For reference, the upstream README's RTX 4090
`Time/Step` is 45.8 ms for pi0.5 and 20.6 ms for FlashVLA.

## Against the official CUDA numbers

README reference (RTX 4090, 2 views, `benchmarks/benchmark_latency.py`, 100 samples after
10 warm-up): pi0.5 **45.8 ms/chunk**, FlashVLA **26.7 ms/chunk** -> **1.72x**.

| device | pi0.5 | FlashVLA | ratio |
|---|---|---|---|
| RTX 4090 | 45.8 ms | 26.7 ms | 1.72x |
| 910B3 | 559 ms | 130 ms | **4.3x** |

- Absolute: FlashVLA on 910B3 is ~**4.9x slower** than on a 4090 (130 vs 26.7 ms);
  pi0.5 is ~**12.2x slower** (559 vs 45.8 ms).
- The algorithmic win is **larger** on Ascend, because one denoise step costs relatively
  more here (no fused attention/MLP kernels) while the prefix forward is paid once per
  chunk either way.
- The paper's `Time/Step` (53.8 ms pi0.5, 22.1 ms d=1, 20.6 ms d=2) includes MuJoCo
  rendering on GPU. Our LIBERO wall-clock numbers below include CPU-only rendering, so
  those rows are not comparable.

## Async overlap under a realistic env step

`AsyncStreamingActionManager` pipelines inference against the environment step: while the
env is stepping on CPU, the next chunk is already being decoded on the NPU. Whether that
hides anything depends entirely on how expensive the env step is, so the manager was
measured twice — with the env step simulated as 0 ms and as 600 ms (`BENCH_ENV_MS`).
600 ms is the LIBERO wall-clock range actually observed in the accuracy runs
(351-623 ms/action step), so it is the realistic case.

Four chips, 120 manager steps per configuration, `n_action_steps=10`, RTC off:

| env step | overlap=0 | overlap=1 | overlap=2 | overlap=3 |
|---|---|---|---|---|
| 0 ms (chip 4) | 12.72 ms | 12.95 | 12.96 | 12.95 |
| 0 ms (chip 7) | 12.02 ms | 11.96 | 12.10 | 12.12 |
| **600 ms (chip 5)** | **613.96 ms** | 613.06 | 612.58 | 612.49 |
| **600 ms (chip 6)** | **612.04 ms** | 612.09 | 612.13 | 612.11 |

Reading of these numbers:

- With a 600 ms env step the action step costs **612-614 ms**, and only **12-13 ms of it
  is the policy** (the `stall` field, i.e. the blocking inference cost, stays at 12.1-13.1
  ms for every overlap setting). The env step is **~98 %** of the step.
- Going from `overlap=0` to `overlap=3` saves **1.5 ms out of 614 ms, i.e. 0.24 %**.
  Async overlap has effectively no payoff on this configuration.
- With a 0 ms env step there is nothing to overlap and overlap>0 is a wash (±0.3 ms).
- This is the same conclusion the LIBERO accuracy runs imply from the other side: 351-623
  ms per action step wall clock against 12-13 ms of policy time.

**Implication:** on a machine whose simulator is CPU-side MuJoCo with software rendering,
overlap cannot be the lever. It becomes worth revisiting only after rendering and
observation preprocessing are fast enough that the env step approaches the ~12 ms policy
step — or on hardware where the policy is the dominant term. This is a statement about
*this* harness, not about the paper's setting, where the GPU env loop and the GPU policy
are much closer in cost.

## Concurrency: four chips at once

The four `A1-A4` latency runs above were launched simultaneously, one per chip, each with
its own `ASCEND_RT_VISIBLE_DEVICES`, its own process and its own 8.4 GB HBM allocation:

| | chips 4,5,6,7 concurrent | same configs, earlier single-chip runs |
|---|---|---|
| action step | 12.23 / 12.85 / 12.38 / 12.73 ms | 12.43 / 13.44 / 12.98 ms |

- Four concurrent runs land inside the single-run spread, so per-chip isolation holds: no
  HCCS contention, no HBM pressure (4 x 8.4 GB of 8 x 64 GB), no run-to-run drift.
- Wall clock for the whole sweep is therefore ~1/4 of running the four sequentially, which
  makes a 4-configuration sweep affordable.
- Caveat: this holds for independent single-chip benchmarks. Anything that uses HCCS
  collectives (multi-card FSDP/training) or fills HBM can still interfere.

## LIBERO accuracy baseline

10 episodes x 10 tasks per suite, seed 1000, synchronous decoding:

| suite | ours (10 ep/task) | paper pi0.5 (2000 ep, 5 runs) | paper FlashVLA d=0 |
|---|---|---|---|
| libero_spatial | **96.0 %** (96/100) | 98.8 | 98.6 |
| libero_object | **99.0 %** (99/100) | 98.2 | 99.0 |
| libero_goal | **99.0 %** (99/100) | 98.0 | 97.8 |
| libero_10 | **96.0 %** (96/100) | 92.4 | 96.2 |
| average | **97.5** | 96.9 | 97.9 |

- Per-suite deltas are within sampling noise for n=100 (±~2 %); the paper averages 2000
  episodes over 5 seeds. `libero_spatial` is the one suite clearly below both paper rows
  (96.0 vs 98.6/98.8) and deserves a longer run before drawing conclusions.
- Per-task successes, spatial / object / goal / libero_10:
  `[10,10,10,10,10,9,10,8,10,10]` / `[10,10,10,9,10,10,10,10,10,10]` /
  `[10,10,10,9,10,10,10,10,10,10]` / `[10,10,10,10,10,10,9,10,8,9]`.

Wall clock of the same runs (environment-bound, see note):

| suite | s / episode | ms / action step | eval wall |
|---|---|---|---|
| libero_spatial | 82.4 | 622.9 | 2 h 35 m |
| libero_object | 56.3 | 351.1 | 1 h 46 m |
| libero_goal | 68.7 | 515.4 | 2 h 10 m |
| libero_10 | 102.0 | 371.5 | 3 h 03 m |

Over 95 % of that wall time is MuJoCo offscreen rendering on CPU (llvmpipe) plus
observation preprocessing; the policy contributes 12-13 ms per step.

## Deliberately not optimised (candidate next steps)

1. **Prefix reuse across chunks** — 56-62 % of chunk time; the model recomputes the
   observation prefix + prefill on every chunk even when the image is unchanged.
2. **`torch.compile`** — unavailable in this environment: `BackendCompilerFailed
   backend='inductor' ... No module named 'triton'`, no Inductor Ascend backend.
3. **Async chunk overlap** — measured at **0.24 %** (614 -> 612 ms/step) as long as the
   environment loop costs ~600 ms/step; it only pays off after rendering/preprocessing
   gets faster. See [Async overlap](#async-overlap-under-a-realistic-env-step).
4. **Batching / multi-chip** — untested; this baseline is batch 1 on one chip.
5. **Startup** — 320 s -> 2.8 s is available, but kept opt-in so the default path stays
   bit-for-bit the upstream one.

## Reproduce

```bash
# policy latency (policy-only + vanilla comparison); one chip at a time
for c in 5 6 7; do
  CARD=$c FLASHVLA_FAST_STARTUP=1 BENCH_CHUNKS=40 BENCH_WARMUP=4 BENCH_STEPS=60 \
  BENCH_OVERLAP=0 BENCH_VANILLA=1 BENCH_VANILLA_ITERS=5 BENCH_TAG=bench_final_c$c \
  bash run_bench.sh
done

# the four-chip sweep behind the A1-A4 latency rows and the overlap tables:
# one process per chip, launched at the same time (see Concurrency above)
for spec in "4 0 phaseA_env0_a" "7 0 phaseA_env0_b" "5 600 phaseA_env600_a" "6 600 phaseA_env600_b"; do
  set -- $spec
  CARD=$1 BENCH_ENV_MS=$2 BENCH_OVERLAP=0,1,2,3 BENCH_STEPS=120 BENCH_CHUNKS=40 \
  BENCH_WARMUP=4 BENCH_VANILLA=1 BENCH_VANILLA_ITERS=5 FLASHVLA_FAST_STARTUP=1 \
  BENCH_TAG=$3 nohup bash run_bench.sh > /home/data/flashvla/logs/phaseA/$3.log 2>&1 &
done
wait

# accuracy (one suite per invocation; run at most 3 concurrently, EGL context limit)
SUITE=libero_spatial EPS=10 CARD=3 TAG=spatial10 bash run_eval_generic.sh
SUITE=libero_object  EPS=10 CARD=4 TAG=object10  bash run_eval_generic.sh
SUITE=libero_goal    EPS=10 CARD=6 TAG=goal10    bash run_eval_generic.sh
SUITE=libero_10      EPS=10 CARD=0 TAG=libero10  bash run_eval_generic.sh
```

`run_bench.sh` / `run_eval_generic.sh` are documented in [docs/ascend.md](ascend.md);
both wrappers live at the repo root and select the chip, the Ascend runtime, the
offline-HF flags and the log location; `bench_npu.py` reads its own `BENCH_*` switches
from the environment.

**Metric definition.** One `predict_action_chunk` call in FlashVLA graduates exactly one
buffer slot, i.e. `C` executable actions — not `n_action_steps` chunks. `Time/Step` is
`mean_ms / n_action_steps`, which equals `mean_ms / C` only because the LIBERO config
happens to have `chunk_size == n_action_steps == 10`. The benchmark now also emits
`time_per_action_ms = mean_ms / chunk_size` and a `time_per_action_step_note` so the
two definitions cannot be confused when the configs diverge.

Raw results: `bench_npu.py` writes `<tag>.json` under
`sim_eval/libero/outputs/eval/<date>/<time>_<job>/`, and the wrapper's log goes to
`$LOG_DIR/<tag>_c<chip>.log`. Accuracy runs write
`/home/data/flashvla/logs/<tag>_out/eval_info.json`.
