"""TPU runtime probe sample.

What it is: a public-safe excerpt of the MegaCpp POC runtime probe that checks
whether `torch_xla` and `libtpu` are actually usable.

Why it exists: importing one TPU module is not enough to prove that the runtime
can execute real graphs.

What problem it solves: it separates "TPU package installed" from "TPU runtime
actually alive" so launchers can choose a fallback path early.
"""

from __future__ import annotations

import importlib
import importlib.util


def has_libtpu_install() -> bool:
    try:
        return importlib.util.find_spec("libtpu") is not None
    except Exception:
        return False


def _probe_ok(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    try:
        return int(value) >= 1
    except Exception:
        return bool(value)


def has_usable_torch_xla_runtime() -> bool:
    try:
        xr = importlib.import_module("torch_xla.runtime")
        for probe_name in (
            "global_runtime_device_count",
            "addressable_runtime_device_count",
            "world_size",
        ):
            probe = getattr(xr, probe_name, None)
            if not callable(probe):
                continue
            try:
                if _probe_ok(probe()):
                    return True
            except Exception:
                continue
    except Exception:
        pass

    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
        for probe_name in ("xrt_world_size", "get_xla_supported_devices"):
            probe = getattr(xm, probe_name, None)
            if not callable(probe):
                continue
            try:
                if _probe_ok(probe()):
                    return True
            except Exception:
                continue
    except Exception:
        pass

    return False


def summarize_tpu_runtime_probe() -> dict[str, object]:
    return {
        "libtpu_installed": has_libtpu_install(),
        "usable_torch_xla_runtime": has_usable_torch_xla_runtime(),
        "message": (
            "A usable runtime needs both the package and at least one working device probe."
        ),
    }
