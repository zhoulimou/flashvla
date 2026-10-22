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

| chip | chunk mean | p50 | p95 | ms / action step | encode | prefill | denoise |
|---|---|---|---|---|---|---|---|
| 5 | 124.3 ms | 123.9 | 132.7 | **12.43** | 34.7 | 40.3 | 46.7 |
| 6 | 134.4 ms | 134.2 | 135.5 | **13.44** | 38.5 | 44.7 | 52.1 |
| 7 | 129.8 ms | 129.1 | 137.9 | **12.98** | 35.8 | 41.7 | 49.3 |

- Working range across runs and chips: **120-134 ms per chunk, 12.0-13.4 ms per action
  step**. The node is shared, so treat a ±5 % spread as noise.
- Observation-prefix work (`encode` + `prefill`) is **56-62 %** of the chunk. That is the
  single largest Ascend-side optimisation target: reusing the prefix across chunks (or a
  cheaper image encoder path) would remove most of it.
- Peak HBM: **8424 MB** per process at batch 1.

## Vanilla pi0.5 decoding, same weights, same chip

The honest question is "what would the original sampler cost here". Two variants:

| variant | chunk latency | ms / action step | FlashVLA speedup |
|---|---|---|---|
| (a) prefix cached once per chunk — this is what the flow-matching sampler really does | 559.1 / 598.7 / 549.8 ms | 55.9 / 59.9 / 55.0 | **4.50 / 4.45 / 4.24x** |
| (b) prefix recomputed on every denoise step | 1246.1 / 1323.4 / 1238.2 ms | 124.6 / 132.3 / 123.8 | 10.02 / 9.85 / 9.54x |

**Headline: 4.2-4.5x.** Variant (b) is the pessimistic figure that an earlier version of
this harness reported (9.5-10x); it overstates the gain because no real pi0.5 sampler
recomputes the prefix on every step.

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
3. **Async chunk overlap** — at most ~2 % headroom as long as the environment loop costs
   ~600 ms/step; it only pays off after rendering/preprocessing gets faster (see
   `docs/ascend.md`).
4. **Batching / multi-chip** — untested; this baseline is batch 1 on one chip.
5. **Startup** — 320 s -> 2.8 s is available, but kept opt-in so the default path stays
   bit-for-bit the upstream one.

## Reproduce

```bash
# policy latency (policy-only + vanilla comparison), three chips, no evals running
for c in 5 6 7; do
  CARD=$c FLASHVLA_FAST_STARTUP=1 BENCH_CHUNKS=40 BENCH_WARMUP=4 BENCH_STEPS=60 \
  BENCH_OVERLAP=0 BENCH_VANILLA=1 BENCH_VANILLA_ITERS=5 BENCH_TAG=bench_final_c$c \
  bash run_bench.sh
done

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

Raw results land in `/home/data/flashvla/logs/bench_npu/*.json` and
`/home/data/flashvla/logs/<tag>_out/eval_info.json`.
