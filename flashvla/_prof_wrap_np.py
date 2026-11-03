"""Per-phase profiling wrapper around flashvla.train.update_policy.

Inserts torch.npu.synchronize() between forward / backward / clip / optimizer /
loss.item() so the async NPU stream cannot hide work in the wrong bucket.
The extra syncs remove some compute/comm overlap, so totals will be slightly
higher than an uninstrumented run -- use them for attribution, not for ETA.

Usage:
  accelerate launch --multi_gpu --num_processes=4 --module flashvla._prof_wrap \
    --config_path=train/configs/pi05/libero/pi05_flashvla_npu_b4_prof.yaml
"""
from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import nullcontext

import torch

import flashvla.npu_support  # noqa: F401
import flashvla.train as T
from flashvla.train import has_method

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
LOG_PATH = os.environ.get("PROF_LOG", "/home/data/flashvla/logs/prof_phases.jsonl")
PROF_DIR = os.environ.get("PROF_TORCH_DIR", "/home/data/flashvla/prof_out")
_STATE = {"micro": 0, "step": 1}
_PROF = {"obj": None, "enabled": os.environ.get("PROF_TORCH") == "1" and RANK == 0}


def _prof_start() -> None:
    """Start the Ascend profiler on the first training step (not during model build)."""
    if not _PROF["enabled"] or _PROF["obj"] is not None:
        return
    import torch_npu

    want_stack = os.environ.get("PROF_STACK") == "1"
    # Stack capture is heavy: keep a single active step so the export stays sane.
    sched = (
        torch_npu.profiler.schedule(wait=2, warmup=1, active=1, repeat=1)
        if want_stack
        else torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=want_stack,
        with_modules=want_stack,
        schedule=sched,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(PROF_DIR),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        ),
    )
    prof.start()
    _PROF["obj"] = prof
    atexit.register(_prof_stop)


def _prof_stop() -> None:
    prof = _PROF["obj"]
    if prof is None:
        return
    _PROF["obj"] = None
    try:
        prof.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"[prof] stop failed: {exc}", flush=True)


def _sync() -> None:
    torch.npu.synchronize()


def _emit(row: dict) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def update_policy_prof(
    train_metrics,
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    accelerator,
    lr_scheduler=None,
    lock=None,
    *,
    loss_scale: float = 1.0,
    do_step: bool = True,
    autocast_ctx=None,
):
    policy.train()
    _prof_start()
    rec = {
        "rank": RANK,
        "step": _STATE["step"],
        "micro": _STATE["micro"],
        "do_step": bool(do_step),
    }

    _sync()
    t0 = time.perf_counter()
    ctx = autocast_ctx if autocast_ctx is not None else accelerator.autocast()
    with ctx:
        loss, output_dict = policy(batch)
        raw_loss = loss.detach()
        loss = loss * loss_scale
    _sync()
    t1 = time.perf_counter()

    accelerator.backward(loss)
    _sync()
    t2 = time.perf_counter()

    t3 = t2
    t4 = t2
    grad_norm_value = None
    if do_step:
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )
        grad_norm_value = grad_norm.item()  # implicit sync point
        t3 = time.perf_counter()

        with lock if lock is not None else nullcontext():
            optimizer.step()

        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
        _sync()
        t4 = time.perf_counter()

    t4b = time.perf_counter()
    train_metrics.loss = raw_loss.item()
    if grad_norm_value is not None:
        train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    t5 = time.perf_counter()

    rec.update(
        fwd=t1 - t0,
        bwd=t2 - t1,
        clip=t3 - t2,
        opt=t4 - t3,
        loss_item=t5 - t4b,
        total=t5 - t0,
    )
    if do_step:
        _STATE["step"] += 1
        _STATE["micro"] = 0
        if _PROF["obj"] is not None:
            _PROF["obj"].step()
    else:
        _STATE["micro"] += 1
    if LOCAL_RANK == 0:
        _emit(rec)
    return train_metrics, output_dict


def main() -> None:
    T.update_policy = update_policy_prof
    T.main()


if __name__ == "__main__":
    main()
