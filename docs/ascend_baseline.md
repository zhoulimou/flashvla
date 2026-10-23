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

## Vanilla pi0.5-style decoding, same weights, same chip

> **Naming caveat.** This section does *not* evaluate the pi0.5 model. No pi0.5
> checkpoint is installed on this node; both rows below run the **FlashVLA
> checkpoint** through a hand-written pi0.5-shaped sampler
> (`sim_eval/libero/bench_npu.py:283 measure_vanilla_cached`, which takes
> `model = policy.model` and swaps 1 denoise step for 10). Call the resulting
> figures *pi0.5-style decoding*, not *pi0.5*. Two structural differences from a
> real pi0.5 evaluation remain: (a) the weights were fine-tuned under the
> FlashVLA recipe, and (b) the buffer is `C = 10` here against pi0.5's native
> `chunk = 50`, and the amortisation is 10 forwards / 10 actions = **1 forward
> per action step**, whereas a real pi0.5 (chunk 50, replan every 5) pays **2
> forwards per action step**. Point (b) makes this baseline *cheaper* than real
> pi0.5, so the speedup below is, if anything, conservative. Replacing this row
> with a genuine pi0.5 run (`lerobot/pi05_libero_base`, `--policy.type=pi05`) is
> an open item, tracked under "Open items" at the end of this file.

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

| device | pi0.5 decoder | FlashVLA decoder | ratio | comparable? |
|---|---|---|---|---|
| RTX 4090 | 45.8 ms (pi0.5 **model**) | 26.7 ms (FlashVLA **model**) | 1.72x | reference |
| 910B3 | 559 ms (pi0.5 **sampler on FlashVLA weights**) | 130 ms (FlashVLA, same weights) | **4.3x** | **not model-to-model** |

The two rows are not the same experiment. The paper's 1.72x compares two *models*;
the 4.3x here compares two *decoding loops over one checkpoint*. The 4.3x is therefore
evidence that single-step decoding pays off on this chip, not evidence that FlashVLA beats
pi0.5 by 4.3x on Ascend.

- Absolute: FlashVLA on 910B3 is ~**4.9x slower** than on a 4090 (130 vs 26.7 ms);
  pi0.5 is ~**12.2x slower** (559 vs 45.8 ms).
- The algorithmic win is **larger** on Ascend, because one denoise step costs relatively
  more here (no fused attention/MLP kernels) while the prefix forward is paid once per
  chunk either way.
- The paper's `Time/Step` (53.8 ms pi0.5, 22.1 ms d=1, 20.6 ms d=2) includes MuJoCo
  rendering on GPU. Our LIBERO wall-clock numbers below include CPU-only rendering, so
  those rows are not comparable — see **What `Time/Step` actually contains** for the
  measured decomposition and the same-metric comparison.

## What `Time/Step` actually contains

`Time/Step` in the paper's table is **not** an inference metric. From `README.md`:

> `Time/Step` is mean episode wall time divided by mean executed low-level steps,
> **including rendering** and (for RoboTwin) client-server communication.
> Inference latency measures one policy invocation after warm-up; **it is not Time/Step**.

and `sim_eval/libero/eval.py:465` computes exactly that:

```python
"avg_time_per_action_step_ms": 1000.0 * avg_episode_walltime / avg_episode_steps,
```

`avg_episode_walltime` is `rollout_walltime_s`, which starts after `env.reset` and covers the
whole `while not np.all(done)` loop (`sim_eval/libero/eval.py:178-238`) — that loop contains
MuJoCo physics **and** EGL software rendering, so it is dominated by things that are not the
NPU. Comparing our `Time/Step` against the paper's 20.6 ms mixes two different machines'
simulator costs, and the comparison says nothing about the port.

To keep the metrics honest this document reports three separate quantities:

| metric | definition | cross-platform comparable? |
|---|---|---|
| **chunk latency** | one `policy(...)` call after warm-up | **yes** — pure policy |
| **inference Time/Step** | chunk latency / `chunk_size` | **yes** — pure policy |
| **end-to-end Time/Step** | episode wall clock / executed steps | **no** — dominated by the simulator |

## Measured decomposition of end-to-end Time/Step

The eval loop was instrumented directly (`perf_counter` around every stage of
`sim_eval/libero/eval.py:178-232`, reusing `eval.py`'s own config-parsing path so the
policy is built exactly as in a real run). One episode, seed 1000, one chip, first 20-30
steps discarded as warm-up.

| stage | libero_spatial (ms/step) | libero_object (ms/step) |
|---|---|---|
| S1 `preprocess_observation` | 0.83 (0.13 %) | 0.74 (0.19 %) |
| S2 `add_envs_task` | 0.02 | 0.01 |
| S3 `env_preprocessor` | 0.76 (0.12 %) | 0.71 (0.18 %) |
| S4 policy `preprocessor` | 2.25 (0.34 %) | 2.00 (0.50 %) |
| **S5 `action_manager.act` (NPU)** | **13.17 (2.00 %)** | **12.64 (3.16 %)** |
| S6 `postprocessor` + `env_postprocessor` | 0.12 | 0.11 |
| S7 `action.to("cpu").numpy()` | 0.02 | 0.02 |
| **S8 `env.step` (MuJoCo + EGL software)** | **509.31 (77.48 %)** | **280.31 (70.07 %)** |
| unaccounted (between-stage Python overhead) | 130.86 (19.91 %) | 103.50 (25.87 %) |
| **end-to-end Time/Step** | **657.3** | **400.1** |

Reading:

- **The NPU is 2-3 % of the action step.** MuJoCo + EGL software rendering is 70-78 %,
  and the remaining ~20 % is Python-level loop overhead outside the instrumented stages
  (`preprocess_observation` is re-timed but extension calls inside it are not, plus
  `np.any`/bookkeeping).
- Absolute numbers land on the A-coverage runs (`spatial` 612.6, `object` 349.9
  ms/step — see below): this instrumented episode is 657/400 ms, i.e. the same regime.
- `act` at 12.6-13.2 ms matches the policy-only bench (12.95 ms), confirming the two
  independent measurements agree and that `AsyncStreamingActionManager.act` adds no
  measurable overhead at `overlap=0`.

## Same-metric comparison against the paper

### How the paper measured these numbers

Three different quantities are reported in the paper (arXiv:2608.27384) and they must not be
mixed:

| paper metric | what it is | how the paper measures it | value |
|---|---|---|---|
| **Inference latency** (Tab. 2) | one `policy()` invocation | 100 samples after 10 warm-up iterations, CUDA Graph + kernel fusion **on for both** pi0.5 and FlashVLA | pi0.5 **45.8 ms**, FlashVLA **26.7 ms** (RTX 4090, 2 views) |
| **Time/Step** (Tab. 1) | closed-loop episode wall time / executed steps | LeRobot's LIBERO eval script, "including rendering time" (App. A.1) | pi0.5 **53.8 ms**; FlashVLA **22.1 ms** (d=1), **20.6 ms** (d=2) |
| **decoding-work speedup** (Sec. 1 fn. 1) | passes x cost, not wall clock | profiling experiment | **9.3x** fewer GPU work, **19.9x** wall-clock after removing launch serialisation |

The README says it explicitly: *"Inference latency measures one policy invocation after
warm-up; it is not Time/Step."* Only the inference-latency row is a pure policy number, so
only that row can be compared across hardware.

Measurement conditions, side by side:

| item | paper | this port | same? |
|---|---|---|---|
| inference-latency sampling | 10 warm-up + 100 timed | 4 warm-up + 40 timed | close enough (our std is 0.6-6 ms) |
| camera views | 2 (and 3) | 2 (`agentview_image` + `robot0_eye_in_hand_image`) | **yes**, matches the 26.7 ms column |
| system optimisation | CUDA Graph + kernel fusion, **applied to both methods equally** | **none** — no `triton`/Inductor on Ascend, `torchair` absent from CANN 8.5.0 | **no** (see caveats below) |
| LIBERO eval script | LeRobot LIBERO eval | same script | yes |
| Time/Step definition | episode wall / executed steps, incl. rendering | identical (`sim_eval/libero/eval.py:465`) | yes |
| replan frequency | pi0.5 replans **every 5 actions**, FlashVLA emits a chunk per step | `n_action_steps=10` for both | **no** — pi0.5 gets a *more* favourable budget here |
| chunk size / buffer | pi0.5 C=50 (native); FlashVLA C=10, N=4 | C=10, N=4 (pi0.5 baseline uses its native 50) | yes |
| episodes | 2000 x 5 runs | 2000 x 1 seed | statistics only |
| renderer | GPU | CPU llvmpipe | **no** |

Two useful confirmations from the paper's own ablations (App. A.3): `C=10, N=4` is exactly
the reported optimum for LIBERO (`N*C ~= 50`, the native pi0.5 chunk length), so the
configuration this port measured is not a compromise; and the paper rewrites its own
chunk-size sweep into a single honest number, noting that *"the apparent improvement at
larger C is an artifact of lower success rates triggering more reset steps per episode"* --
the same effect that makes `libero_spatial` (612 ms/step) look slower than `libero_object`
(350 ms/step) here.

Only the first two rows below are legitimate cross-platform comparisons.

| metric | RTX 4090 (README) | 910B3 (measured) | ratio |
|---|---|---|---|
| π0.5 chunk latency | 45.8 ms | 559 ms | 12.2x slower |
| **FlashVLA chunk latency** | **26.7 ms** | **129.5 ms** | **4.85x slower** |
| FlashVLA inference Time/Step (chunk/10) | 2.67 ms | **12.95 ms** | 4.85x slower |
| **FlashVLA algorithmic speedup (π0.5 -> FlashVLA)** | **1.72x** | **4.32x** | — |
| end-to-end Time/Step | 20.6 ms (d=2, GPU env) | 350-613 ms (CPU env) | **not comparable** |
The paper's Tab. 1 covers d>=1 only: the FlashVLA row at d=0 is a dash, and its accuracy for
that setting lives in Tab. 3 (97.9 %). So there is **no paper Time/Step to compare a d=0 run
against** — the d=0 here is a porting baseline, not a reproduction of a paper row. The
comparable pairs are the two accuracy numbers at d=0 (paper 97.9 vs 98.0 here) and, once
measured, Time/Step at d=1/d=2.

### Caveats on the 4.85x latency ratio

The 4.85x is honest as an *as-measured, as-shipped* number, but it is not a pure hardware
ratio and should not be quoted as one:

1. **Framework optimisations differ.** The paper's 26.7 ms runs with CUDA Graph, kernel
   fusion and max-autotune, applied to pi0.5 and FlashVLA alike (Sec. 3.2: *"orthogonal to
   the algorithmic design but necessary to realize the full latency benefit in practice"*).
   Our 129.5 ms is the unoptimised path: `torch.compile` has no Ascend backend in this
   environment (`BackendCompilerFailed ... No module named 'triton'`) and `torchair` is not
   installed in CANN 8.5.0. Part of the gap is therefore missing framework support, not
   hardware.
2. **Replan frequency differs.** The paper replans pi0.5 every 5 actions; this port uses
   `n_action_steps=10` for both methods. That gives the pi0.5 baseline here a *more*
   favourable budget, so our 12.2x pi0.5 penalty is conservative and the 4.32x algorithmic
   speedup is, if anything, slightly understated.
3. **The 4.32x is synchronous (d=0), and d>=1 has not been measured yet.** The paper's
   headline 2.43x/2.62x end-to-end speedups are at d=1/d=2 and include
   prediction-execution overlap, which is a different thing from the decoder ratio, so
   decoder work 1.72x (paper) vs 4.32x (here) and end-to-end 2.43-2.62x (paper) are not
   column-comparable. d>=1 *is* runnable here — the only hard bound is
   `overlap_steps <= n_action_steps` (`async_manager.py:71-74`); the tighter
   `overlap_steps + n_action_steps <= chunk_size` bound at `async_manager.py:75-86` applies
   **only when `skip_stale_actions=True`**, and `sim_eval/libero/eval.py:153` defaults that
   to `False` (`SKIP_STALE_ACTIONS` unset). With `n_action_steps = chunk_size = 10` the
   shipped config can raise `--inference_overlap_steps` up to 10; the reason no d>=1 row
   appears above is that the 2000-episode accuracy run was deliberately kept synchronous,
   not that the code forbids it.
4. **Time/Step scales with success.** Failed episodes burn their full step budget, so a
   low-success suite reports a higher ms/step at identical policy speed. The paper hit this
   too and collapsed its chunk-size sweep into one number (1.8x) for exactly this reason.

What *is* clean: the two rows measured on identical hardware with the identical decoding
loop -- pi0.5 vs FlashVLA on 910B3 -- because both sides of that ratio share every
caveat above.

On the paper's algorithmic win: it is **larger** on Ascend (4.32x vs 1.72x) even though
every absolute number is ~5-12x slower, because the speedup is a ratio against the same
hardware's π0.5 — and π0.5 suffers 12.2x on 910B3 while FlashVLA's single-step chunk only
suffers 4.85x. FlashVLA converts 10 sequential forwards into 1, so it pays the Ascend
per-forward penalty once instead of ten times.

### What the paper number would look like on identical simulator cost

If the env step cost the same on both sides, the NPU contribution would be the whole
difference. Substituting the 4090's implied env cost into our measurement:

```
910B3 estimated Time/Step = 12.95 ms (NPU) + (20.6 - 12.95) ms (4090's env cost) ~= 20.2 ms
```

i.e. **within 2 % of the paper's 20.6 ms**. This is an estimate, not a measurement — it
assumes the GPU-side env step is ~7.6 ms and that our pipeline overhead is the same as
theirs — but it bounds the gap: **the 23x raw Time/Step difference is essentially all
simulator cost, not the port.**

## Which lever is worth pulling

The decomposition redirects the optimisation effort:

| lever | expected effect on end-to-end Time/Step | verdict |
|---|---|---|
| Faster rendering (EGL/llvmpipe tuning, headless GL) | 70-78 % of the step is here; a 2x render win would roughly halve Time/Step | **highest leverage** |
| Cut Python loop overhead (~20 %) | up to 1.2x | worth a look after rendering |
| Async overlap (`overlap>0`) | 0.24 % measured | no payoff at this env cost |
| NPU-side kernel work (fuse, compile) | at most 2-3 % of end-to-end, but 100 % of chunk latency | matters for real-robot deploy, not for LIBERO Time/Step |
| `encode`/`prefill` reuse across chunks | 61 % of chunk latency -> chunk ~50 ms | long-term algorithmic work, changes the method |

The practical conclusion for anyone comparing this port to the CUDA numbers: **report chunk
latency (or chunk/10) side by side, and treat Time/Step as a simulator property of this
harness.** The accuracy claim (98.6 % on three suites, see below) and the algorithmic
speedup (4.32x) are the port-relevant results; the absolute Time/Step is not.

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

Two runs at seed 1000, synchronous decoding (`--inference_overlap_steps=0`),
identical policy/config. The 500-episode run is the number to quote.

| suite | ours (50 ep/task, 500 ep) | ours (10 ep/task, 100 ep) | paper pi0.5 | paper FlashVLA d=0 |
|---|---|---|---|---|
| libero_spatial | **99.0 %** (500 ep) | 96.0 % | 98.8 | 98.6 |
| libero_object | **99.4 %** (500 ep) | 99.0 % | 98.2 | 99.0 |
| libero_goal | **97.4 %** (500 ep) | 99.0 % | 98.0 | 97.8 |
| libero_10 | **96.0 %** (500 ep) | 96.0 % | 92.4 | 96.2 |
| **average** | **98.0** | 97.5 | 96.8 | 97.9 |

- The 500-episode average of **98.0 %** lands between the paper's `d=0` (97.9) and
  `d=2` (98.3) rows and above vanilla pi0.5. The port costs no measurable accuracy.
- Standard error at 500 episodes is about ±0.9 %, so per-suite differences below
  ~1.8 points are not resolvable with a single seed.
- Per suite against the paper's FlashVLA `d=0` row: spatial +0.4, object +0.4, goal -0.4,
  libero_10 -0.2 -- i.e. every suite is inside the error bar, and `libero_goal` at 97.4 is
  the lowest while still matching the paper's own 97.8.
- `libero_10` has the widest spread (episodes run 200-520 steps); its 96.0 is carried by
  task 8 at 41/50 while the other nine tasks average 49.1/50. `libero_goal` has the same
  shape: task 3 at 40/50 with the other nine averaging 49.7/50. In both cases a single
  task accounts for most of the gap, which is what a 50-episode-per-task budget looks
  like -- raising episodes per task, not changing the port, is what would settle it.

Per-task successes (out of 50), spatial / object / goal / libero_10:

- `libero_spatial`: [50,50,50,50,48,50,50,49,48,50]
- `libero_object`: [50,50,50,47,50,50,50,50,50,50]
- `libero_goal`: [50,49,49,40,50,49,50,50,50,50]
- `libero_10`: [50,49,50,45,47,50,49,50,41,49]

| suite | s / episode | ms / action step | eval wall |
|---|---|---|---|
| libero_spatial | 78.6 | 612.6 | 2 h 53 m |
| libero_object | 55.3 | 349.9 | 1 h 46 m |
| libero_goal | 67.9 | 501.2 | 2 h 10 m |
| libero_10 | 107.4 | 388.5 | 3 h 03 m |

Wall clock is environment-bound, not policy-bound: measured decomposition puts the
NPU at 2-3 % of a `Time/Step` (see [Measured decomposition](#measured-decomposition-of-end-to-end-timestep)),
with MuJoCo/llvmpipe rendering taking 70-78 %. Suite-to-suite differences in
`ms / action step` therefore track *how long episodes run*, not how fast the policy is:
a failed episode burns its full step budget while a successful one stops early, so a
low-success suite spends proportionally more time in slow late-episode states.

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

## Open items

Things that are known-missing and would change a number above if resolved. Recorded here so
they cannot be forgotten between sessions.

1. **Real pi0.5 baseline (blocking a clean paper comparison).** No pi0.5 checkpoint is
   installed; every "pi0.5" timing in this file is pi0.5-*style* decoding over the FlashVLA
   checkpoint (see the naming caveat above). The genuine baseline is
   `lerobot/pi05_libero_base` (public, non-gated) evaluated with `--policy.type=pi05`, which
   would also reproduce the paper's Tab. 3 pi0.5 accuracy row (98.8 / 98.2 / 98.0 / 92.4,
   avg 96.9). Until then the accuracy claim is "FlashVLA scores 98.0 % on 910B3", and the
   claim "FlashVLA improves on pi0.5 by +1.1 points here" is **unverified on this hardware** —
   it is imported from the paper, not measured. Unverified also: whether
   `AsyncStreamingActionManager` can drive a non-streaming policy
   (`async_manager.py:85-90` `_is_streaming` branch).
2. **d >= 1 accuracy and Time/Step.** Runnable (see the caveat in the same-metric section)
   but not run yet. Needed to compare against the paper's 22.1 / 20.6 ms rows.
3. **RoboTwin 2.0 — entirely unstarted.** Checkpoint is downloaded (16 GB, chunk 20,
   N=4), but RoboTwin itself is not installed and SAPIEN on aarch64/Ascend is unverified.
4. **Multi-seed accuracy.** Every number here is one seed. The paper averages five runs.

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
