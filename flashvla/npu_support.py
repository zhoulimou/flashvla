"""Ascend NPU support shims for lerobot's device helpers.

lerobot's device whitelist (``lerobot.utils.device_utils`` and
``lerobot.configs.policies``) only knows cuda / mps / xpu / cpu, so
``--policy.device=npu`` is rejected during CLI parsing, before the model is
ever built. Importing this module teaches those helpers about the ``npu``
backend provided by ``torch_npu``. It is a no-op on hosts without an Ascend
stack (torch then simply never auto-loads ``torch_npu``).
"""

from __future__ import annotations

import logging

import torch

_log = logging.getLogger(__name__)

_NPU_AVAILABLE = False
try:  # torch auto-loads torch_npu when the CANN toolkit is sourced
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = bool(torch.npu.is_available())
except Exception as exc:  # pragma: no cover - non-NPU hosts
    _log.debug("torch_npu unavailable: %s", exc)


def is_npu_device(device: object) -> bool:
    return isinstance(device, str) and device.split(":")[0] == "npu"


def npu_available() -> bool:
    return _NPU_AVAILABLE


def apply() -> None:
    """Patch lerobot's device helpers in place (idempotent)."""
    if not _NPU_AVAILABLE:
        return
    try:
        from lerobot.utils import device_utils as du
    except Exception:  # pragma: no cover
        return
    if getattr(du, "_flashvla_npu_patched", False):
        return

    orig_is_available = du.is_torch_device_available
    orig_auto_select = du.auto_select_torch_device
    orig_is_amp = du.is_amp_available

    def is_torch_device_available(try_device: str) -> bool:
        if is_npu_device(try_device):
            return torch.npu.is_available()
        return orig_is_available(try_device)

    def auto_select_torch_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if npu_available():
            return torch.device("npu")
        return orig_auto_select()

    def is_amp_available(device: str) -> bool:
        if is_npu_device(device):
            return True
        return orig_is_amp(device)

    du.is_torch_device_available = is_torch_device_available
    du.auto_select_torch_device = auto_select_torch_device
    du.is_amp_available = is_amp_available
    du._flashvla_npu_patched = True

    # ``lerobot.configs.policies`` imported the helpers by value.
    for mod_name in ("lerobot.configs.policies", "lerobot.policies.factory"):
        try:
            mod = __import__(mod_name, fromlist=["_"])
        except Exception:
            continue
        if hasattr(mod, "is_torch_device_available"):
            mod.is_torch_device_available = is_torch_device_available
        if hasattr(mod, "auto_select_torch_device"):
            mod.auto_select_torch_device = auto_select_torch_device
        if hasattr(mod, "is_amp_available"):
            mod.is_amp_available = is_amp_available


apply()
