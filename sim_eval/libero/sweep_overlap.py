#!/usr/bin/env python
"""Sweep ``inference_overlap_steps`` on real LIBERO rollouts (Ascend/NPU friendly).

Loads the policy once, builds the envs once, then replays the same task(s) with
different chunk-overlap settings. This is the only way to measure the actual
benefit of FlashVLA's asynchronous manager: without an environment in the loop
the policy is the only thing running, so overlap has nothing to hide
(see ``bench_npu.py`` phase 4).

Usage (same CLI as ``eval.py``; extra knobs are env vars)::

    SWEEP_OVERLAPS=0,1,2,3 SWEEP_TAG=sweep_spatial0 \
    python sim_eval/libero/sweep_overlap.py \
        --policy.path=.../flashvla-pi05-libero --policy.device=npu \
        --env.type=libero --env.task=libero_spatial --env.task_ids="[0]" \
        --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=5 \
        --output_dir=/home/data/flashvla/logs/sweep_spatial0

Results: ``<output_dir>/<SWEEP_TAG>.json`` (written after every overlap value)
plus a one-line summary per run on stdout.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_libero_eval_module():
    """Import ``eval.py`` without running its ``__main__`` block."""
    spec = importlib.util.spec_from_file_location("libero_eval", _HERE / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["libero_eval"] = module
    spec.loader.exec_module(module)
    return module


ev = _load_libero_eval_module()

import torch  # noqa: E402
from lerobot.configs import parser  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

from flashvla.policies.factory import make_policy, make_pre_post_processors  # noqa: E402


def _overlaps() -> list[int]:
    raw = os.environ.get("SWEEP_OVERLAPS", "0,1,2,3")
    return [int(x) for x in raw.replace(" ", "").split(",") if x != ""]


def _scalar_metrics(metrics: dict) -> dict:
    """Flatten a per-task metrics dict into plain JSON scalars."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
            out[key] = float(sum(value)) / len(value)
        elif isinstance(value, list) and value and all(isinstance(v, bool) for v in value):
            out[key] = float(sum(value)) / len(value)
    return out


@parser.wrap()
def sweep_main(cfg: ev.AsyncEvalPipelineConfig):
    init_logging()
    overlaps = _overlaps()
    tag = os.environ.get("SWEEP_TAG", "overlap_sweep")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = ev.get_safe_torch_device(cfg.policy.device)
    print(f"[sweep] tag={tag} overlaps={overlaps} n_episodes={cfg.eval.n_episodes} device={device}")

    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    results = {
        "tag": tag,
        "device": str(device),
        "n_episodes": cfg.eval.n_episodes,
        "seed": cfg.seed,
        "runs": {},
    }

    with torch.no_grad():
        for overlap in overlaps:
            t0 = time.time()
            info = ev.eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                inference_overlap_steps=overlap,
                max_episodes_rendered=0,
                videos_dir=out_dir / "videos",
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
                only_record_failures=True,
            )
            wall = time.time() - t0
            overall = {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in info["overall"].items()
                if k != "video_paths"
            }
            per_task = [
                {
                    "task_group": task["task_group"],
                    "task_id": task["task_id"],
                    "metrics": _scalar_metrics(task["metrics"]),
                }
                for task in info["per_task"]
            ]
            results["runs"][str(overlap)] = {
                "overall": overall,
                "per_task": per_task,
                "walltime_s": wall,
            }
            (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=2))
            print(
                f"[sweep] overlap={overlap}: pc_success={overall.get('pc_success')} "
                f"episode_s={overall.get('avg_episode_walltime_s')} "
                f"ms_per_action_step={overall.get('avg_time_per_action_step_ms')} "
                f"eval_s={overall.get('eval_s')} wall={wall:.1f}s"
            )

    summary = {k: v["overall"] for k, v in results["runs"].items()}
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sweep] wrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    sweep_main()
