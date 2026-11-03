"""Split a LIBERO suite eval into per-task shards, then merge them back losslessly.

Why: one eval.py process runs a suite's 10 tasks sequentially on a single chip,
so `libero_10` alone needs ~16 h. Spreading the 40 (suite, task) shards over
three machines' chips cuts the wall clock to a few hours.

Correctness: `eval.py` derives episode seeds only from `--seed` and the batch
index inside `eval_one`, never from which tasks are in the suite. A shard for
task k therefore runs exactly the episodes the full-suite run would have run,
so merging the raw `successes` lists reproduces the full-suite numbers.

Usage:
    # build shard list
    python eval_shard.py plan --out /home/data/flashvla/logs/prod50k/plan.txt

    # run one shard (called by the dispatcher)
    python eval_shard.py run --suite libero_spatial --task 3 --policy <ckpt> \
        --out /home/data/flashvla/logs/prod50k/shard_out --chip 0

    # merge all shard outputs
    python eval_shard.py merge --shards /home/data/flashvla/logs/prod50k/shard_out \
        --out /home/data/flashvla/logs/prod50k/merged_eval_info.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASKS_PER_SUITE = 10


def cmd_plan(args: argparse.Namespace) -> int:
    rows = [f"{s} {t}" for s in SUITES for t in range(TASKS_PER_SUITE)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} shards to {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run one (suite, task) shard on one chip.

    The Ascend toolkit must be sourced or `import torch` cannot load torch_npu,
    and LIBERO needs MUJOCO_GL=egl with a real (non-software) EGL platform. Doing
    that here, in a bash wrapper, keeps every shard identical to what
    run_precision_npu.sh does for a full-suite run.
    """
    out = Path(args.out) / f"{args.suite}_task{args.task}"
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    cli = " ".join([
        f'"{py}"', "-u", "sim_eval/libero/eval.py",
        f"--policy.path={args.policy}",
        "--policy.device=npu",
        "--policy.compile_model=false",
        "--policy.fuse_qkv=false",
        "--policy.fuse_gate_up=false",
        f"--policy.n_action_steps={args.n_action_steps}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        f"--env.task_ids=[{args.task}]",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        f"--eval.n_episodes={args.n_episodes}",
        f"--inference_overlap_steps={args.overlap}",
        f"--seed={args.seed}",
        f'--output_dir="{out}"',
    ])
    script = (
        "set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; set -u; "
        f"export ASCEND_RT_VISIBLE_DEVICES={args.chip}; "
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "
        "PYTHONUNBUFFERED=1 FLASHVLA_FAST_STARTUP=1 MUJOCO_GL=egl "
        "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; "
        "unset EGL_PLATFORM LIBGL_ALWAYS_SOFTWARE; "
        f"exec {cli}"
    )
    print(f"[shard] {args.suite} task={args.task} chip={args.chip} -> {out}", flush=True)
    rc = subprocess.call(["/bin/bash", "-c", script], cwd="/home/z00633863/flashvla")
    print(f"[shard] {args.suite} task={args.task} rc={rc}", flush=True)
    return rc


def _agg(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


ACC = ("sum_rewards", "max_rewards", "successes", "episode_steps",
       "episode_walltimes_s", "video_paths")


def _summarise(acc: dict) -> dict:
    return {
        "avg_sum_reward": _agg(acc["sum_rewards"]),
        "avg_max_reward": _agg(acc["max_rewards"]),
        "pc_success": _agg(acc["successes"]) * 100 if acc["successes"] else float("nan"),
        "avg_episode_steps": _agg(acc["episode_steps"]),
        "avg_episode_walltime_s": _agg(acc["episode_walltimes_s"]),
        "avg_time_per_action_step_ms": (
            1000.0 * _agg(acc["episode_walltimes_s"]) / _agg(acc["episode_steps"])
        ),
        "n_episodes": len(acc["sum_rewards"]),
        "video_paths": list(acc["video_paths"]),
    }


def cmd_merge(args: argparse.Namespace) -> int:
    root = Path(args.shards)
    infos = sorted(root.rglob("eval_info.json"))
    if not infos:
        print(f"no eval_info.json under {root}", file=sys.stderr)
        return 1

    per_task: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for path in infos:
        d = json.loads(path.read_text())
        for rec in d.get("per_task", []):
            key = (rec["task_group"], int(rec["task_id"]))
            if key in seen:
                print(f"[merge] duplicate {key} from {path}, skipping", file=sys.stderr)
                continue
            seen.add(key)
            per_task.append(rec)

    per_task.sort(key=lambda r: (SUITES.index(r["task_group"])
                                 if r["task_group"] in SUITES else 99, r["task_id"]))

    group_acc: dict[str, dict[str, list]] = {}
    overall: dict[str, list] = {k: [] for k in ACC}
    for rec in per_task:
        g = rec["task_group"]
        m = rec["metrics"]
        gacc = group_acc.setdefault(g, {k: [] for k in ACC})
        for k in ACC:
            v = m.get(k) or []
            if k == "video_paths":
                gacc[k].extend(v);
                overall[k].extend(v)
            else:
                gacc[k].extend(v);
                overall[k].extend(v)

    per_group = {g: _summarise(acc) for g, acc in group_acc.items()}
    overall_sum = _summarise(overall)
    # eval_s is wall clock, meaningless after sharding; report the max shard time
    eval_s = 0.0
    for path in infos:
        d = json.loads(path.read_text())
        eval_s = max(eval_s, float(d.get("overall", {}).get("eval_s", 0.0) or 0.0))
    overall_sum["eval_s"] = eval_s
    overall_sum["eval_ep_s"] = eval_s / max(1, overall_sum["n_episodes"])

    merged = {"per_task": per_task, "per_group": per_group, "overall": overall_sum}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(merged, indent=2))

    print(f"[merge] {len(per_task)} task records from {len(infos)} shard dirs")
    for g, s in per_group.items():
        print(f"  {g:<16} pc_success={s['pc_success']:.1f}  n={s['n_episodes']}")
    print(f"  {'OVERALL':<16} pc_success={overall_sum['pc_success']:.2f}  "
          f"n={overall_sum['n_episodes']}")
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan"); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run")
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chip", type=int, required=True)
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--seed", type=int, default=1000)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("merge")
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
