#!/usr/bin/env python
"""NPU latency benchmark for FlashVLA (pi0.5) single-stream decoding.

Measures, on one Ascend NPU chip, the numbers that matter for the
"Time/Step" comparison in the FlashVLA paper (π0.5 53.8 ms -> FlashVLA
20.6 ms on GPU):

  1. ``policy_only``: wall-clock latency of one ``predict_action_chunk``
     call in steady streaming (1 denoise step -> C executable actions),
     divided by ``n_action_steps``. This is the paper's decoder metric:
     it excludes environment simulation and rendering entirely.
  2. ``phases``: encode / prefill / denoise split of the same chunk,
     measured by re-running the steady-streaming computation with
     explicit device synchronisation around each stage.
  3. ``vanilla``: the pi0.5 decoding strategy the paper compares against —
     ``denoise_steps`` (10) full forwards per chunk, i.e. prefix encode +
     prefill recomputed on every denoise step (the original lerobot sampler
     keeps no cross-step KV cache). Time/Step = chunk_ms / n_action_steps,
     which is the same definition as ``policy_only``, so the two numbers
     form a same-device speedup ratio.
  4. ``manager``: per-env-step latency of ``AsyncStreamingActionManager``
     for ``inference_overlap_steps`` in {0,1,2,3}. Overlap>0 pipelines
     inference against the (CPU-bound) env step; the mean of this
     distribution is the achievable Time/Step once the env loop is
     overlapped, the tail is the residual blocking cost.

The policy and the observation are built exactly the way
``sim_eval/libero/eval.py`` builds them, so the timings are comparable to
the rollout profile in the evaluation logs.

Example (Ascend):
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_RT_VISIBLE_DEVICES=5 MUJOCO_GL=egl HF_HUB_OFFLINE=1
    python sim_eval/libero/bench_npu.py \
        --policy.path=/home/data/flashvla/weights/flashvla-pi05-libero \
        --policy.device=npu --policy.n_action_steps=10 \
        --env.type=libero --env.task=libero_spatial \
        --env.max_parallel_tasks=1 --eval.batch_size=1 \
        --output_dir=/home/data/flashvla/logs/bench_npu

Env-var knobs (kept out of the draccus schema so the same CLI flags as
eval.py keep working):
    BENCH_CHUNKS=50    steady chunk samples for phase 1/2
    BENCH_WARMUP=5     warmup chunks before timing
    BENCH_STEPS=120    manager steps per overlap setting
    BENCH_OVERLAP=0,1,2,3
    BENCH_VANILLA=1    run phase 3 (vanilla 10-step decode); 0 to skip
                       3a = prefix cached once per chunk (the honest baseline),
                       3b = prefix recomputed every denoise step
    BENCH_DENOISE_STEPS=10  denoise steps of the vanilla sampler
    BENCH_ENV_MS=0     simulated env-step cost (ms) per step in phase 4
    BENCH_TAG=bench    output json stem
"""

import contextlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import parser
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

import flashvla.npu_support  # noqa: F401  (patches lerobot device whitelist for npu)
from flashvla.async_manager import AsyncStreamingActionManager
from flashvla.policies.factory import make_policy, make_pre_post_processors
from flashvla.startup import no_random_init
from flashvla.policies.pi0.utils import build_attention_mask_and_position_ids


def _load_libero_eval_module():
    """Reuse eval.py's config + device helper without duplicating them."""
    path = Path(__file__).resolve().parent / "eval.py"
    spec = importlib.util.spec_from_file_location("_libero_eval_for_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_eval_mod = _load_libero_eval_module()
AsyncEvalPipelineConfig = _eval_mod.AsyncEvalPipelineConfig
get_safe_torch_device = _eval_mod.get_safe_torch_device


@dataclass
class BenchConfig(AsyncEvalPipelineConfig):
    """Eval config plus benchmark-only flags."""


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


_AMP: bool = False


def _amp_ctx():
    """bf16 autocast for npu, or a no-op (keeps params fp32, casts compute)."""
    if not _AMP:
        return contextlib.nullcontext()
    return torch.autocast(device_type="npu", dtype=torch.bfloat16)


def _timed(device: torch.device) -> float:
    _sync(device)
    return perf_counter() * 1000.0


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    }


def _peak_mem_mb(device: torch.device) -> float | None:
    try:
        if device.type == "npu":
            return torch.npu.max_memory_allocated() / (1024**2)
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:  # noqa: BLE001 - best effort only
        return None
    return None


def build_obs(cfg: BenchConfig):
    """Create one real LIBERO observation, preprocessed like eval.py does."""
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs,
                    trust_remote_code=cfg.trust_remote_code)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    first_group = next(iter(envs))
    first_task = next(iter(envs[first_group]))
    env = envs[first_group][first_task]
    obs, _ = env.reset(seed=cfg.seed)
    obs = preprocess_observation(obs)
    obs = add_envs_task(env, obs)
    obs = env_preprocessor(obs)
    return envs, obs


def instrumented_chunk(policy, batch: dict, device: torch.device) -> dict:
    """One steady-streaming chunk with a device-synced stage breakdown.

    Mirrors ``PI05FlashVLAModel._steady_streaming`` exactly, but brackets
    each stage with ``_sync`` so the ms numbers are real device time
    instead of host launch time. Slower than the uninstrumented path by
    design (the extra syncs forbid pipelining between stages).
    """
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    N, C = model.N, model.C
    buf_len = N * C
    bsz = tokens.shape[0]
    dev = tokens.device
    action_dim = model.config.max_action_dim
    buffer = model.sample_noise((bsz, buf_len, action_dim), dev)

    t0 = _timed(device)
    ctx = _amp_ctx()
    ctx.__enter__()
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.prefix_embedder(
        images, img_masks, tokens, masks
    )
    t1 = _timed(device)

    for layer in model.layers:
        layer.self_attn.attn.reset_cache()
    prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
        prefix_pad_masks, prefix_att_masks, prefix_embs.dtype,
    )
    hidden = [prefix_embs, None]
    conds = [None, None]
    for layer in model.layers:
        hidden = layer(hidden, prefix_attention_mask, prefix_position_ids, conds, use_cache=True)
    t2 = _timed(device)

    time_per_slot = torch.arange(1, N + 1, device=dev, dtype=torch.float32) / N
    time = time_per_slot.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)
    v_t = model.denoise_step(prefix_pad_masks, prefix_att_masks, state, buffer, time, padding_mask=None)
    ctx.__exit__(None, None, None)
    t3 = _timed(device)

    # keep the same buffer update as the real path so the following
    # uninstrumented calls stay numerically on-distribution
    dt = 1.0 / N
    buffer = buffer - dt * v_t
    model.sample_noise(buffer.shape, dev)
    del hidden, conds

    return {
        "encode_ms": t1 - t0,
        "prefill_ms": t2 - t1,
        "denoise_ms": t3 - t2,
        "total_ms": t3 - t0,
    }


def measure_chunks(policy, batch, device, n_chunks: int, n_warmup: int) -> dict:
    """Time N steady-streaming ``predict_action_chunk`` calls (paper metric)."""
    policy.reset()
    n_action_steps = policy.config.n_action_steps
    warm_lat: list[float] = []
    lat: list[float] = []
    with torch.inference_mode():
        for _ in range(n_warmup):
            policy.predict_action_chunk(batch)
        _sync(device)
        for _ in range(n_chunks):
            start = perf_counter() * 1000.0
            ctx = _amp_ctx()
            ctx.__enter__()
            policy.predict_action_chunk(batch)
            ctx.__exit__(None, None, None)
            _sync(device)
            warm_lat.append(perf_counter() * 1000.0 - start)
    lat = warm_lat
    stats = _stats(lat)
    stats["n_action_steps"] = int(n_action_steps)
    stats["chunk_size"] = int(policy.model.C)
    stats["time_per_action_step_ms"] = stats["mean_ms"] / n_action_steps
    stats["time_per_action_step_median_ms"] = stats["median_ms"] / n_action_steps
    return stats


def measure_phases(policy, batch, device, n: int) -> dict:
    per_stage: dict[str, list[float]] = {}
    with torch.inference_mode():
        for _ in range(n):
            bd = instrumented_chunk(policy, batch, device)
            for k, v in bd.items():
                per_stage.setdefault(k, []).append(v)
    out = {k: _stats(v) for k, v in per_stage.items()}
    total = out.get("total_ms", {}).get("mean_ms", 0.0)
    if total:
        for k, s in out.items():
            if k != "total_ms":
                s["pct_of_total"] = 100.0 * s["mean_ms"] / total
    return out



def measure_vanilla_cached(
    policy, batch: dict, device: torch.device, n: int, denoise_steps: int
) -> dict:
    """Vanilla pi0.5 decoding with the prefix computed *once* per chunk.

    This is how the original flow-matching sampler runs: ``prefix_embedder`` + prefill
    once, then ``denoise_steps`` KV-cached ``denoise_step`` calls.  It is the honest
    baseline for "how much does FlashVLA save", because FlashVLA also pays the prefix
    exactly once per chunk and only replaces the 10 denoise steps with 1.
    """
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    C = model.C
    bsz = tokens.shape[0]
    dev = tokens.device
    action_dim = model.config.max_action_dim
    dt = 1.0 / denoise_steps
    chunks: list[float] = []

    with torch.inference_mode():
        for _ in range(n):
            x = model.sample_noise((bsz, C, action_dim), dev)
            ctx = _amp_ctx()
            ctx.__enter__()
            start = perf_counter() * 1000.0
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.prefix_embedder(
                images, img_masks, tokens, masks
            )
            for layer in model.layers:
                layer.self_attn.attn.reset_cache()
            prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
                prefix_pad_masks, prefix_att_masks, prefix_embs.dtype,
            )
            hidden = [prefix_embs, None]
            conds = [None, None]
            for layer in model.layers:
                hidden = layer(hidden, prefix_attention_mask, prefix_position_ids, conds, use_cache=True)
            for step in range(denoise_steps):
                t = torch.full((bsz, C), step * dt, device=dev, dtype=torch.float32)
                v_t = model.denoise_step(
                    prefix_pad_masks, prefix_att_masks, state, x, t, padding_mask=None
                )
                x = x - dt * v_t
            ctx.__exit__(None, None, None)
            _sync(device)
            chunks.append(perf_counter() * 1000.0 - start)

    stats = _stats(chunks)
    n_action_steps = policy.config.n_action_steps
    stats["denoise_steps"] = int(denoise_steps)
    stats["n_action_steps"] = int(n_action_steps)
    stats["prefix_computed_per_chunk"] = 1
    stats["time_per_action_step_ms"] = stats["mean_ms"] / n_action_steps
    return stats


def measure_vanilla(policy, batch: dict, device: torch.device, n: int, denoise_steps: int) -> dict:
    """Vanilla pi0.5 decoding: ``denoise_steps`` full forwards per chunk.

    Each denoise step recomputes the prefix and prefill exactly like the
    original lerobot sampler (no KV cache reuse across steps), which is
    what the FlashVLA paper's 53.8 ms Time/Step baseline measures.
    Returns per-chunk and per-action-step latency.
    """
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    C = model.C
    bsz = tokens.shape[0]
    dev = tokens.device
    action_dim = model.config.max_action_dim
    dt = 1.0 / denoise_steps
    chunks: list[float] = []
    stages: dict[str, list[float]] = {"encode_ms": [], "prefill_ms": [], "denoise_ms": []}

    with torch.inference_mode():
        for _ in range(n):
            x = model.sample_noise((bsz, C, action_dim), dev)
            start = perf_counter() * 1000.0
            ctx = _amp_ctx()
            ctx.__enter__()
            for step in range(denoise_steps):
                prefix_embs, prefix_pad_masks, prefix_att_masks = model.prefix_embedder(
                    images, img_masks, tokens, masks
                )
                for layer in model.layers:
                    layer.self_attn.attn.reset_cache()
                prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
                    prefix_pad_masks, prefix_att_masks, prefix_embs.dtype,
                )
                hidden = [prefix_embs, None]
                conds = [None, None]
                for layer in model.layers:
                    hidden = layer(hidden, prefix_attention_mask, prefix_position_ids, conds, use_cache=True)
                t = torch.full((bsz, C), step * dt, device=dev, dtype=torch.float32)
                v_t = model.denoise_step(prefix_pad_masks, prefix_att_masks, state, x, t, padding_mask=None)
                x = x - dt * v_t
            ctx.__exit__(None, None, None)
            _sync(device)
            chunks.append(perf_counter() * 1000.0 - start)

    stats = _stats(chunks)
    n_action_steps = policy.config.n_action_steps
    stats["denoise_steps"] = int(denoise_steps)
    stats["n_action_steps"] = int(n_action_steps)
    stats["time_per_action_step_ms"] = stats["mean_ms"] / n_action_steps
    return stats


def measure_manager(policy, batch, device, overlap: int, n_steps: int, env_ms: float = 0.0) -> dict:
    """Per-env-step latency of the async overlap manager.

    ``env_ms`` simulates CPU-side env work (MuJoCo step + render) that runs
    while the NPU chunk inference is in flight; with ``overlap_steps > 0``
    that work is what the pipelining hides behind.
    """
    n_action_steps = policy.config.n_action_steps
    mgr = AsyncStreamingActionManager(policy=policy, overlap_steps=overlap, skip_stale_actions=False)
    mgr.reset()
    lat: list[float] = []
    env_sleep = max(0.0, env_ms) / 1000.0
    with torch.inference_mode():
        for _ in range(n_steps):
            start = perf_counter() * 1000.0
            mgr.act(batch)
            if env_sleep > 0:
                # busy-ish sleep: CPU env work does not touch the NPU queue
                time.sleep(env_sleep)
            lat.append(perf_counter() * 1000.0 - start)
    # drop cold start + first chunk transition
    skip = 3 * n_action_steps
    lat = lat[skip:]
    stats = _stats(lat)
    if lat:
        arr = np.asarray(lat)
        stats["fraction_blocking"] = float((arr > 0.5 * np.percentile(arr, 95)).mean())
        stats["max_ms"] = float(arr.max())
        stats["policy_stall_ms"] = float(stats["mean_ms"] - env_ms)
    stats["overlap_steps"] = int(overlap)
    stats["sim_env_ms"] = float(env_ms)
    return stats



@parser.wrap()
def bench_main(cfg: BenchConfig):
    init_logging()
    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device)
    n_chunks = int(os.environ.get("BENCH_CHUNKS", "50"))
    n_warmup = int(os.environ.get("BENCH_WARMUP", "5"))
    n_steps = int(os.environ.get("BENCH_STEPS", "120"))
    overlaps = [int(x) for x in os.environ.get("BENCH_OVERLAP", "0,1,2,3").split(",") if x.strip() != ""]
    env_ms = float(os.environ.get("BENCH_ENV_MS", "0"))
    run_vanilla = os.environ.get("BENCH_VANILLA", "1") != "0"
    denoise_steps = int(os.environ.get("BENCH_DENOISE_STEPS", "10"))
    tag = os.environ.get("BENCH_TAG", "bench")
    amp_modes = [m.strip() for m in os.environ.get("BENCH_AMP", "0").split(",") if m.strip()]

    print(f"[bench] device={device} chunks={n_chunks} warmup={n_warmup} "
          f"steps={n_steps} overlaps={overlaps} sim_env_ms={env_ms}")

    envs, obs = build_obs(cfg)
    print("[bench] observation ready, loading policy (this can take minutes)...")
    # Optional start-up shortcut (does not change the measured regions or the weights):
    # skipping the discarded random init turns a 320 s build into ~3 s.
    _fast_startup = os.environ.get("FLASHVLA_FAST_STARTUP", "0") == "1"
    with no_random_init() if _fast_startup else contextlib.nullcontext():
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    with torch.inference_mode():
        batch = preprocessor(obs)
    batch = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
    _sync(device)
    print(f"[bench] policy on {next(policy.parameters()).device}, batch keys={sorted(batch)}")

    # -- warm the CANN op compiler with one full chunk cycle
    with torch.inference_mode():
        for _ in range(4):
            policy.predict_action_chunk(batch)
    _sync(device)
    try:
        if device.type == "npu":
            torch.npu.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass

    results: dict = {
        "device": str(device),
        "chip": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "policy_path": str(cfg.policy.pretrained_path),
        "n_action_steps": int(policy.config.n_action_steps),
        "num_buffer_slots": int(policy.model.N),
        "chunk_size": int(policy.model.C),
        "dtype": str(next(policy.parameters()).dtype),
    }

    global _AMP
    results["variants"] = {}
    for mode in amp_modes:
        _AMP = mode not in ("0", "fp32", "off", "False", "false")
        label = "bf16_autocast" if _AMP else "fp32"
        print(f"[bench] variant {label}: policy-only + phases")
        var: dict = {}
        var["policy_only"] = measure_chunks(policy, batch, device, n_chunks, n_warmup)
        var["phases"] = measure_phases(policy, batch, device, max(3, min(10, n_chunks)))
        results["variants"][label] = var
        print(f"  {label}: chunk {var['policy_only']['mean_ms']:.1f} ms -> "
              f"Time/Step {var['policy_only']['time_per_action_step_ms']:.2f} ms | "
              f"encode {var['phases']['encode_ms']['mean_ms']:.1f} prefill "
              f"{var['phases']['prefill_ms']['mean_ms']:.1f} denoise "
              f"{var['phases']['denoise_ms']['mean_ms']:.1f}")
    _AMP = False
    first = results["variants"][next(iter(results["variants"]))]
    results["policy_only"] = first["policy_only"]
    results["phases"] = first["phases"]

    if env_ms is not None and run_vanilla:
        vanilla_iters = int(os.environ.get("BENCH_VANILLA_ITERS", "5"))
        print("[bench] phase 3a: vanilla 10-step decode, prefix cached once per chunk")
        results["vanilla"] = measure_vanilla_cached(
            policy, batch, device, vanilla_iters, denoise_steps
        )
        print(json.dumps(results["vanilla"], indent=2))
        print("[bench] phase 3b: vanilla with prefix recomputed on every denoise step")
        results["vanilla_recompute"] = measure_vanilla(
            policy, batch, device, vanilla_iters, denoise_steps
        )
        print(json.dumps(results["vanilla_recompute"], indent=2))
        if results["policy_only"].get("time_per_action_step_ms"):
            sp_cached = results["vanilla"]["time_per_action_step_ms"] / results["policy_only"]["time_per_action_step_ms"]
            sp_recomp = results["vanilla_recompute"]["time_per_action_step_ms"] / results["policy_only"]["time_per_action_step_ms"]
            results["speedup_vs_vanilla_cached"] = round(sp_cached, 2)
            results["speedup_vs_vanilla_recompute"] = round(sp_recomp, 2)
            print(f"[bench] FlashVLA vs vanilla (cached prefix):   {sp_cached:.2f}x")
            print(f"[bench] FlashVLA vs vanilla (recomputed prefix): {sp_recomp:.2f}x")

    print("[bench] phase 4: async overlap manager step latency")
    results["manager"] = {}
    for ov in overlaps:
        results["manager"][str(ov)] = measure_manager(policy, batch, device, ov, n_steps, env_ms)
        st = results["manager"][str(ov)]
        print(f"  overlap={ov}: mean={st['mean_ms']:.2f} ms/step p95={st['p95_ms']:.2f} "
              f"max={st['max_ms']:.2f} stall={st.get('policy_stall_ms', 0):.2f}")

    results["peak_hbm_mb"] = _peak_mem_mb(device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"BENCH RESULTS ({device}, chip={results['chip']})")
    print("=" * 72)
    po = results["policy_only"]
    print(f"  chunk latency: mean {po['mean_ms']:.1f} ms | p50 {po['median_ms']:.1f} | "
          f"p95 {po['p95_ms']:.1f}")
    print(f"  Time/Step    : {po['time_per_action_step_ms']:.2f} ms "
          f"(chunk/n_action_steps)   [paper GPU: pi0.5 53.8 / FlashVLA 20.6]")
    va = results.get("vanilla")
    if va:
        speedup = va["time_per_action_step_ms"] / po["time_per_action_step_ms"]
        print(f"  vanilla Time/Step: {va['time_per_action_step_ms']:.2f} ms "
              f"({va['denoise_steps']} denoise) -> FlashVLA speedup {speedup:.2f}x")
    ph = results["phases"]
    print(f"  phases(mean) : encode {ph['encode_ms']['mean_ms']:.1f} | "
          f"prefill {ph['prefill_ms']['mean_ms']:.1f} | "
          f"denoise {ph['denoise_ms']['mean_ms']:.1f} ms")
    for ov, st in results["manager"].items():
        print(f"  overlap={ov:>2}: {st['mean_ms']:6.2f} ms/step  p95 {st['p95_ms']:6.2f}  "
              f"blocking-step frac {st.get('fraction_blocking', 0):.2f}")
    if results["peak_hbm_mb"]:
        print(f"  peak HBM     : {results['peak_hbm_mb']:.0f} MB")
    print(f"  json         : {out_path}")
    print("=" * 72)
    close_envs(envs)


if __name__ == "__main__":
    bench_main()
