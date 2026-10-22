#!/usr/bin/env python
"""Collect ``eval_info.json`` / ``eval_info.partial.json`` files into one table.

Usage::

    python sim_eval/libero/summarize_eval.py /home/data/flashvla/logs/*_out
    python sim_eval/libero/summarize_eval.py /home/data/flashvla/logs --markdown

For every directory it prints the suite, number of episodes, success rate,
average episode length and the wall-clock time per action step (the number that
actually matters when MuJoCo runs in the same process as the policy).
"""

import json
import sys
from pathlib import Path


def _load(directory: Path) -> tuple[dict | None, str]:
    for name in ("eval_info.json", "eval_info.partial.json"):
        path = directory / name
        if path.exists():
            try:
                return json.loads(path.read_text()), name
            except json.JSONDecodeError:
                continue
    return None, ""


def _fmt(value, spec: str) -> str:
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    markdown = "--markdown" in argv
    roots = [Path(a) for a in args] or [Path("/home/data/flashvla/logs")]

    rows = []
    for root in roots:
        dirs = [root] if (root / "eval_info.json").exists() or (root / "eval_info.partial.json").exists() else []
        if not dirs:
            dirs = sorted(p for p in root.iterdir() if p.is_dir())
        for directory in dirs:
            info, source = _load(directory)
            if info is None:
                continue
            overall = info.get("overall") or {}
            groups = info.get("per_group") or {}
            if not overall and groups:
                overall = next(iter(groups.values()))
            tasks = len({t["task_id"] for t in info.get("per_task", []) if t.get("task_id") is not None})
            rows.append(
                {
                    "dir": directory.name,
                    "source": source,
                    "suites": ",".join(sorted(groups)) or "-",
                    "tasks": tasks,
                    "episodes": overall.get("n_episodes"),
                    "pc_success": overall.get("pc_success"),
                    "episode_steps": overall.get("avg_episode_steps"),
                    "episode_s": overall.get("avg_episode_walltime_s"),
                    "ms_per_step": overall.get("avg_time_per_action_step_ms"),
                    "eval_s": overall.get("eval_s"),
                }
            )

    if not rows:
        print("no eval_info.json found", file=sys.stderr)
        return 1

    if markdown:
        header = (
            "| run | suite | tasks | episodes | pc_success | episode_steps | episode_s | ms/action_step | eval_s |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        body = "".join(
            f"| {r['dir']} | {r['suites']} | {r['tasks']} | {r['episodes']} | "
            f"{_fmt(r['pc_success'], '.1f')}% | {_fmt(r['episode_steps'], '.1f')} | "
            f"{_fmt(r['episode_s'], '.1f')} | {_fmt(r['ms_per_step'], '.1f')} | {_fmt(r['eval_s'], '.0f')} |\n"
            for r in rows
        )
        print(header + body)
    else:
        for r in rows:
            print(
                f"{r['dir']:<28} {r['suites']:<22} tasks={r['tasks']:<3} eps={r['episodes']:<4} "
                f"success={_fmt(r['pc_success'], '.1f')}% steps={_fmt(r['episode_steps'], '.1f')} "
                f"t/ep={_fmt(r['episode_s'], '.1f')}s ms/step={_fmt(r['ms_per_step'], '.1f')} ({r['source']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
