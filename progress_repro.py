#!/usr/bin/env python3
"""Progress report for the paper-reproduction precision runs.

For every suite it prints:
  * how many tasks finished (from eval_info.partial.json / eval_info.json), and
  * the in-flight episode counter of the current task (parsed from the log tail),
so you can see the difference between "first task, 46/50 episodes" and
"nine tasks done".

Handles all three directory layouts used by this project:

  paperE/<run>/<ov>/seed<N>/<suite>/eval_info*.json   (d=1, d=2, v044, base per-suite)
  baseE/<ov>/seed<N>/<suite>/eval_info*.json          (no <run> level)
  coverageA/A_<suite>_ep<N>_s<S>_out/eval_info*.json  (flat, one suite per dir)

Usage: python progress_repro.py [ROOT ...]
"""
import glob
import json
import os
import re
import sys

ROOTS = sys.argv[1:] or ["/home/data/flashvla/logs/paperE"]
TASKS_PER_SUITE = 10
EPISODES_PER_TASK = 50

BATCH_RE = re.compile(r"Stepping through eval batches:.*?\|\s*(\d+)/(\d+)\s*\[")
ROLLOUT_RE = re.compile(r"Running rollout with at most.*?\|\s*(\d+)/(\d+)\s*\[")
FLAT_RE = re.compile(r"^(?P<prefix>.*?)(?P<suite>libero_[a-z0-9_]+?)_ep(?P<eps>\d+)_s(?P<seed>\d+)(?P<suffix>_out)?$")


def _find_log(suite_dir):
    """eval.sh writes the log next to the suite directory, not inside it."""
    parent = os.path.dirname(suite_dir)
    base = os.path.basename(suite_dir)
    candidates = [
        os.path.join(parent, base + ".log"),
        os.path.join(parent, re.sub(r"_out$", "", base) + ".log"),
        suite_dir + ".log",
        os.path.join(suite_dir, "*.log"),
    ]
    for c in candidates:
        logs = glob.glob(c)
        if logs:
            return logs[0]
    return None


def _tail(path, limit=400000):
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - limit))
        return fh.read().decode("utf-8", "replace")


def read_progress(suite_dir):
    """Return (label, tasks_done, cur_batch, cur_total, success)."""
    fin = os.path.join(suite_dir, "eval_info.json")
    par = os.path.join(suite_dir, "eval_info.partial.json")
    tasks_done = 0
    success = None
    if os.path.exists(fin):
        try:
            d = json.load(open(fin))
            o = d.get("overall", {})
            return ("FINAL", TASKS_PER_SUITE, None, None, o.get("pc_success"))
        except Exception:
            pass
    if os.path.exists(par):
        try:
            d = json.load(open(par))
            tasks_done = d.get("tasks_done", 0)
            success = d.get("pc_success_so_far")
        except Exception:
            pass

    # in-flight episode counter: last "Stepping through" line in the log.
    log = _find_log(suite_dir)
    cur = tot = None
    if log:
        try:
            tail = _tail(log)
            m = None
            for m in BATCH_RE.finditer(tail):
                pass
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
            else:
                for m in ROLLOUT_RE.finditer(tail):
                    pass
                if m:
                    # rollout progress is steps within one episode; use it only as a
                    # crude "episode in flight" hint, never as an episode count.
                    cur, tot = 0, None
        except Exception:
            pass
    return ("running", tasks_done, cur, tot, success)


def discover(root):
    """Yield (group_label, suite_label, suite_dir) for every suite under root."""
    root = os.path.abspath(root)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 4:
            dirnames[:] = []
            continue
        base = os.path.basename(dirpath)
        has_json = "eval_info.json" in filenames or "eval_info.partial.json" in filenames
        # A suite that has not finished its first task yet has no eval_info*.json,
        # but eval.sh still writes <suite>.log next to it and creates videos/.
        has_log = os.path.exists(os.path.join(os.path.dirname(dirpath), base + ".log"))
        if has_json or (has_log and os.path.isdir(os.path.join(dirpath, "videos"))):
            m = FLAT_RE.match(base)
            if m and not rel.count(os.sep):
                # coverageA-style flat layout: one suite per top-level directory.
                group = f"ep{m.group('eps')} seed{m.group('seed')}"
                suite = m.group("suite")
            else:
                suite = base
                parent = os.path.dirname(rel)
                group = parent if parent != "." else "(root)"
            found.append((group, suite, dirpath))
            dirnames[:] = []  # do not descend into videos/ etc.
    return found


def main():
    for root in ROOTS:
        rows = []
        for group, suite, suite_dir in discover(root):
            state, done, cur, tot, succ = read_progress(suite_dir)
            eps_done = done * EPISODES_PER_TASK + (cur or 0)
            eps_total = TASKS_PER_SUITE * EPISODES_PER_TASK
            pct = 100.0 * eps_done / eps_total
            if state == "FINAL":
                txt = f"FINAL {succ:.1f}%" if succ is not None else "FINAL"
            else:
                txt = f"{eps_done:4d}/{eps_total} ep ({pct:4.1f}%)"
                if succ is not None:
                    txt += f"  {succ:.1f}%"
            rows.append((group, suite, txt))
        if not rows:
            continue
        print(f"### {root}")
        last = None
        for group, suite, txt in sorted(rows):
            if group != last:
                print(f"  [{group}]")
                last = group
            print(f"    {suite:30s} {txt}")
        print()


if __name__ == "__main__":
    main()
