from mechanica import (
    native_kernels_available,
    native_kernels_status,
    native_spring_available,
    native_spring_status,
)
from mechanica import _native


def test_native_kernel_status_reports_loader_failure(monkeypatch) -> None:
    def fail_load():
        raise _native.NativeExtensionUnavailable("native unavailable in test")

    monkeypatch.setattr(_native, "_load_spring_extension", fail_load)

    available, reason = native_kernels_status()

    assert not available
    assert reason == "native unavailable in test"
    assert not native_kernels_available()
    assert native_spring_status() == (False, "native unavailable in test")
    assert not native_spring_available()


def test_native_kernel_status_reports_loader_success(monkeypatch) -> None:
    monkeypatch.setattr(_native, "_load_spring_extension", lambda: object())

    available, reason = native_kernels_status()

    assert available
    assert reason is None
    assert native_kernels_available()
    assert native_spring_status() == (True, None)
    assert native_spring_available()
