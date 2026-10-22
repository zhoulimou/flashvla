"""Regression test for the Ascend device shim: apply/unpatch must round-trip."""
import pytest

from flashvla import npu_support


def _du():
    from lerobot.utils import device_utils as du

    return du


def test_shim_installed_on_import():
    assert getattr(_du(), "_flashvla_npu_patched", False) is True


def test_npu_device_recognised():
    assert npu_support.is_npu_device("npu")
    assert npu_support.is_npu_device("npu:3")
    assert not npu_support.is_npu_device("cuda:0")
    assert not npu_support.is_npu_device(None)


def test_amp_not_advertised_on_npu():
    du = _du()
    if not npu_support.npu_available():
        pytest.skip("no Ascend device on this host")
    assert du.is_amp_available("npu") is False
    assert du.is_amp_available("npu:0") is False


def test_auto_select_returns_npu_when_no_cuda():
    du = _du()
    if not npu_support.npu_available():
        pytest.skip("no Ascend device on this host")
    assert du.auto_select_torch_device().type == "npu"


def test_unpatch_restores_originals_and_is_idempotent():
    du = _du()
    npu_support.unpatch()

    # Originals are back: lerobot's stock helper raises on "npu" again.
    assert du._flashvla_npu_patched is False
    with pytest.raises(ValueError, match="Unknown device npu"):
        du.is_torch_device_available("npu")

    # Calling unpatch() again must be a harmless no-op.
    before = du.is_torch_device_available
    npu_support.unpatch()
    assert du.is_torch_device_available is before

    # apply() can reinstall the shim, and the mirrored modules follow.
    npu_support.apply()
    assert du._flashvla_npu_patched is True
    if npu_support.npu_available():
        assert du.is_torch_device_available("npu") is True
        assert du.is_amp_available("npu") is False

    from lerobot.configs import policies as pol

    if hasattr(pol, "is_torch_device_available") and npu_support.npu_available():
        assert pol.is_torch_device_available("npu") is True


def test_unpatch_leaves_no_stale_state():
    du = _du()
    npu_support.unpatch()
    assert du._flashvla_npu_patched is False
    npu_support.apply()
    assert du._flashvla_npu_patched is True
