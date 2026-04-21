"""CUDA graph env defaults for compiled training blocks.

What it is: a MegaCpp POC-based excerpt of the runtime env defaults that turn on
Inductor CUDA graph trees for compiled CUDA regions.

Why it exists: the training stack needed a graph path that removes Python
dispatch overhead without capturing the whole FSDP2-wrapped model.

What problem it solves: it enables per-block CUDA graph capture for
regional-compile and leaf-compile paths, which avoids the hook-ordering issues
that broke whole-model graph capture while still improving steady-state
throughput where the repository documented it.
"""

from __future__ import annotations

import os


def apply_cuda_graph_env_defaults() -> dict[str, tuple[str, str]]:
    """Apply MegaCpp POC-faithful CUDA graph env defaults unless the user opted out.

    Grounding:
    - The MegaCpp compiled-training path documents these two env vars as the
      PyTorch-native CUDA-graph lane for compiled blocks.
    - The same public sample family records a measured H200:8 gain for this path
      on a regional-compile lane.
    """
    if os.environ.get("PUBLIC_SAMPLE_NO_ENV_DEFAULTS") == "1":
        return {}

    defaults: dict[str, str] = {
        # MegaCpp POC note: compiled regions are captured as CUDA graphs to remove
        # Python dispatch overhead. The MegaCpp POC repo records a +4.5% H200:8
        # throughput gain on one regional-compile FSDP lane.
        "TORCHINDUCTOR_TRITON_CUDAGRAPHS": "1",
        # Required alongside TORCHINDUCTOR_TRITON_CUDAGRAPHS in the MegaCpp POC path.
        "TORCH_COMPILE_CUDAGRAPH_TREES": "1",
    }

    status: dict[str, tuple[str, str]] = {}
    for var, default_value in defaults.items():
        existing = os.environ.get(var)
        if existing is None:
            os.environ[var] = default_value
            status[var] = ("set_by_sample", default_value)
        else:
            status[var] = ("user_provided", existing)
    return status


if __name__ == "__main__":
    for name, (source, value) in apply_cuda_graph_env_defaults().items():
        print(f"{name}={value} ({source})")
