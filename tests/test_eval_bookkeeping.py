"""Tests for the LIBERO evaluation loop's result bookkeeping.

These drive ``eval_policy_all`` with fake environments so they need no MuJoCo,
no EGL context and no Ascend device -- they are meant to run in CI and to catch
the class of bug that actually cost a 50-episode run: the per-task partial
writer was given a new parameter without updating its call sites, so every call
raised ``TypeError`` at bind time, the writer's own try/except could not catch
it, and the completed results were never persisted.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy


class _FakeConfig(PreTrainedConfig):
    """Concrete config so ``_FakePolicy`` satisfies ``PreTrainedPolicy``'s checks."""

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        pass

    @property
    def action_feature(self):
        return SimpleNamespace(shape=(self._fake_action_dim,))

    @property
    def type(self) -> str:
        # ``AsyncStreamingActionManager`` keys off this to pick the streaming path.
        return "pi05-flashvla"
from sim_eval.libero.eval import eval_policy_all


class _FakeVecEnv(gym.vector.VectorEnv):
    """Minimal stand-in for a LIBERO vector env: uniform episodes, fixed length."""

    def __init__(self, steps_per_episode: int = 3, success: bool = True):
        self.num_envs = 1
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1, 7), dtype=np.float32)
        self.metadata = {"render_fps": 30}
        self._steps_per_episode = steps_per_episode
        self._success = success
        self._t = 0
        # lerobot's env helpers reach through ``env.envs[0]`` for task metadata.
        self.envs = [SimpleNamespace(task_description="do the thing", task="do the thing")]

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return ({"dummy": np.zeros((1, 1), dtype=np.float32)}, {})

    def step(self, action):
        self._t += 1
        terminated = self._t >= self._steps_per_episode
        return (
            {"dummy": np.zeros((1, 1), dtype=np.float32)},
            np.array([1.0 if terminated else 0.0], dtype=np.float32),
            np.array([terminated]),
            np.array([False]),
            {"final_info": {"is_success": np.array([terminated])}},
        )

    def call(self, name, *args, **kwargs):
        if name == "_max_episode_steps":
            return [10_000]
        if name == "success":
            return np.array([[self._success]])
        if name in ("task_description", "task"):
            return ["do the thing"]
        raise AttributeError(name)

    def close(self):
        pass


class _FakePipeline:
    """Stands in for lerobot's ``PolicyProcessorPipeline``."""

    def __call__(self, x):
        return x

    def reset(self):
        pass


class _FakePolicy(PreTrainedPolicy):
    """Real ``PreTrainedPolicy`` subclass so ``eval_policy``'s isinstance gate passes."""

    name = "fake"
    config_class = _FakeConfig

    def __init__(self, action_dim: int = 7):
        cfg = _FakeConfig()
        cfg.n_action_steps = 10
        cfg.chunk_size = 10
        cfg._fake_action_dim = action_dim
        super().__init__(config=cfg)
        # ``AsyncStreamingActionManager`` reads the device off the first parameter.
        self._anchor = nn.Parameter(torch.zeros(1))
        # ``[B, n_action_steps, action_dim]`` — the manager indexes chunk[:, step, :].
        self._action = torch.zeros(1, 10, action_dim)

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def get_optim_params(self):
        return []

    def predict_action_chunk(self, batch):
        return self._action

    def select_action(self, batch):
        return self._action

    def reset(self):
        pass

    def forward(self, batch, reduction="mean"):
        return self._action, {}


def _run(tmp_path: Path, *, n_tasks=3, episodes=2, max_parallel_tasks=1, fail_writes=False):
    envs = {"suite_a": {tid: _FakeVecEnv() for tid in range(n_tasks)}}
    pipe = _FakePipeline()
    return eval_policy_all(
        envs=envs,
        policy=_FakePolicy(),
        env_preprocessor=pipe,
        env_postprocessor=pipe,
        preprocessor=pipe,
        postprocessor=pipe,
        n_episodes=episodes,
        start_seed=1000,
        max_parallel_tasks=max_parallel_tasks,
        output_dir=tmp_path,
    )


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_eval_loop_completes_and_aggregates(tmp_path, max_parallel_tasks):
    info = _run(tmp_path, max_parallel_tasks=max_parallel_tasks)

    assert len(info["per_task"]) == 3
    assert info["overall"]["n_episodes"] == 6
    assert info["overall"]["pc_success"] == pytest.approx(100.0)


def test_partial_results_are_written_per_task(tmp_path):
    """Regression: the partial writer used to raise TypeError at every call."""
    _run(tmp_path)

    partial = tmp_path / "eval_info.partial.json"
    assert partial.exists(), "partial results were never persisted"

    payload = json.loads(partial.read_text())
    assert payload["tasks_done"] == 3
    assert payload["tasks_total"] == 3
    assert payload["n_episodes"] == 6
    assert payload["pc_success_so_far"] == pytest.approx(100.0)
    assert len(payload["per_task"]) == 3

    jsonl = tmp_path / "eval_info.partial.jsonl"
    assert jsonl.exists()
    assert len(jsonl.read_text().strip().splitlines()) == 3


def test_bookkeeping_failure_does_not_abort_the_evaluation(tmp_path, monkeypatch):
    """A broken writer must not take down a multi-hour run (call-site guard)."""
    orig_write_text = Path.write_text

    def _boom(self, *args, **kwargs):
        if self.name.endswith(".json.tmp"):
            raise RuntimeError("disk on fire")
        return orig_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _boom)
    info = _run(tmp_path)

    assert len(info["per_task"]) == 3
    assert info["overall"]["n_episodes"] == 6
