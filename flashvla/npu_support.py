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

# Modules that imported lerobot's device helpers *by value* and therefore keep
# private copies after ``lerobot.utils.device_utils`` is patched.
_MIRRORED_MODULES = ("lerobot.configs.policies", "lerobot.policies.factory")
_MIRRORED_NAMES = (
    "is_torch_device_available",
    "auto_select_torch_device",
    "is_amp_available",
)

# Saved originals, populated by ``apply`` and cleared by ``unpatch``.
_ORIGINALS: dict[str, object] = {}


def is_npu_device(device: object) -> bool:
    return isinstance(device, str) and device.split(":")[0] == "npu"


def npu_available() -> bool:
    return _NPU_AVAILABLE


def _repoint_mirrors(replacements: dict[str, object]) -> None:
    """Rewrite the by-value copies in the modules that imported them."""
    for mod_name in _MIRRORED_MODULES:
        try:
            mod = __import__(mod_name, fromlist=["_"])
        except Exception:  # pragma: no cover - optional modules
            continue
        for name, value in replacements.items():
            if hasattr(mod, name):
                setattr(mod, name, value)


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
    _ORIGINALS.update(
        is_torch_device_available=orig_is_available,
        auto_select_torch_device=orig_auto_select,
        is_amp_available=orig_is_amp,
    )

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
        # Empirically, ``torch.autocast("npu", bfloat16)`` is *slower* than
        # plain inference on 910B3 (measured 256.0 vs 134.2 ms/chunk), so AMP
        # must not be advertised as available: lerobot uses this helper to
        # auto-enable AMP, which would silently halve throughput. Note this
        # reports "not worth using", not "not supported" -- ``torch.autocast``
        # itself does work on npu.
        if is_npu_device(device):
            return False
        return orig_is_amp(device)

    replacements = {
        "is_torch_device_available": is_torch_device_available,
        "auto_select_torch_device": auto_select_torch_device,
        "is_amp_available": is_amp_available,
    }
    for name, value in replacements.items():
        setattr(du, name, value)
    du._flashvla_npu_patched = True
    _repoint_mirrors(replacements)


def unpatch() -> None:
    """Restore lerobot's original device helpers (undo :func:`apply`).

    Needed when a single process has to fall back to the stock device path,
    e.g. a test that exercises both branches. No-op if :func:`apply` never
    ran, and safe to call repeatedly.
    """
    try:
        from lerobot.utils import device_utils as du
    except Exception:  # pragma: no cover
        return
    if not getattr(du, "_flashvla_npu_patched", False):
        return

    for name, value in _ORIGINALS.items():
        setattr(du, name, value)
    du._flashvla_npu_patched = False
    _repoint_mirrors(dict(_ORIGINALS))
    _ORIGINALS.clear()


apply()
