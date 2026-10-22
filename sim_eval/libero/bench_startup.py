#!/usr/bin/env python3
"""Measure policy construction ("Making policy.") cost with and without random init.

LeRobot's ``from_pretrained`` builds the policy from its config (randomly initialising
every weight) and only then loads the safetensors checkpoint.  On pi0.5 (3B) that
discarded initialisation dominates start-up time on the Ascend hosts.  ``flashvla.startup``
wraps transformers' ``no_init_weights`` to skip it; this script measures both paths on the
same machine and verifies that the resulting weights are identical to the checkpoint.

Run on the cluster (after sourcing CANN):

    ASCEND_RT_VISIBLE_DEVICES=6 python sim_eval/libero/bench_startup.py \
        --policy.path=/home/data/flashvla/weights/flashvla-pi05-libero \
        --policy.device=npu --output_dir=/home/data/flashvla/logs/startup_out
"""

import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from lerobot.configs import parser
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

import flashvla.npu_support  # noqa: F401  (registers the npu backend for lerobot)
from flashvla.policies.factory import make_policy
from flashvla.startup import no_random_init

import bench_npu as _bn  # same directory; reuses the eval config + device helper

BenchConfig = _bn.BenchConfig
get_safe_torch_device = _bn.get_safe_torch_device
_peak_mem_mb = _bn._peak_mem_mb


def _empty_cache() -> None:
    try:
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache clearing is best effort
        pass


def _build(cfg, fast: bool):
    """Construct the policy; returns (policy, wall seconds)."""
    ctx = no_random_init() if fast else nullcontext()
    _empty_cache()
    t0 = time.perf_counter()
    with ctx:
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    return policy, time.perf_counter() - t0


def _ckpt_path(policy_dir: Path) -> Path | None:
    single = policy_dir / "model.safetensors"
    if single.is_file():
        return single
    sharded = policy_dir / "model.safetensors.index.json"
    if sharded.is_file():
        return sharded
    return None


def _verify_weights(policy, policy_dir: Path, n_samples: int = 8) -> dict:
    """Check that parameters came from the checkpoint, not from uninitialised memory."""
    from safetensors import safe_open

    path = _ckpt_path(policy_dir)
    if path is None:
        return {"checked": False, "reason": "no safetensors found"}

    params = dict(policy.named_parameters())
    report: dict = {"checked": True, "n_params": len(params)}
    if path.name.endswith(".index.json"):
        import json as _json

        index = _json.loads(path.read_text())
        ckpt_keys: set[str] = set(index["weight_map"].keys())
    else:
        with safe_open(str(path), framework="pt") as f:
            ckpt_keys = set(f.keys())

    missing = sorted(k for k in params if k not in ckpt_keys and not k.startswith("model."))
    prefixed_missing = sorted(k for k in params if k not in ckpt_keys)
    report["params_not_in_checkpoint"] = len(prefixed_missing)
    report["params_not_in_checkpoint_sample"] = prefixed_missing[:5]
    report["params_without_model_prefix_not_in_ckpt"] = len(missing)

    matched = [k for k in params if k in ckpt_keys]
    report["params_matched"] = len(matched)
    samples = matched[:n_samples]
    max_diffs = []
    if not path.name.endswith(".index.json"):
        with safe_open(str(path), framework="pt") as f:
            for name in samples:
                ref = f.get_tensor(name).to(torch.float32)
                got = params[name].detach().to("cpu", torch.float32)
                if ref.shape != got.shape:
                    max_diffs.append(float("inf"))
                    continue
                max_diffs.append(float((ref - got).abs().max()))
    report["sampled_max_abs_diff"] = max_diffs
    report["weights_match_checkpoint"] = bool(max_diffs) and max(max_diffs) == 0.0
    return report


@parser.wrap()
def bench_startup(cfg: BenchConfig):
    init_logging()
    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device)
    policy_dir = Path(cfg.policy.pretrained_path)
    print(f"policy: {policy_dir} on {device}")

    results: dict = {"policy_path": str(policy_dir), "device": str(device)}

    policy, slow_s = _build(cfg, fast=False)
    results["baseline"] = {"build_s": round(slow_s, 2), "peak_hbm_mb": _peak_mem_mb(device)}
    print(f"baseline build: {slow_s:.1f}s")
    del policy
    gc.collect()

    policy, fast_s = _build(cfg, fast=True)
    results["no_init"] = {"build_s": round(fast_s, 2), "peak_hbm_mb": _peak_mem_mb(device)}
    print(f"no-init build: {fast_s:.1f}s")

    results["speedup"] = round(slow_s / fast_s, 1) if fast_s > 0 else None
    results["verification"] = _verify_weights(policy, policy_dir)
    print("verification:", json.dumps(results["verification"]))

    out_dir = Path(cfg.output_dir) if getattr(cfg, "output_dir", None) else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bench_startup.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    bench_startup()
