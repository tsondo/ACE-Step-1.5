"""Opt-in deterministic tensor tracing for diffusion debugging.

Set ``ACESTEP_DIFFUSION_TRACE`` to a JSONL output path to record compact,
content-addressed tensor summaries without changing sampler values.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

TRACE_PATH_ENV = "ACESTEP_DIFFUSION_TRACE"

_WRITE_LOCK = threading.Lock()


def _to_numpy(tensor: Any):
    """Convert a PyTorch, MLX, or NumPy tensor to a CPU NumPy array."""
    import numpy as np

    value = tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except (TypeError, RuntimeError):
            pass
    return np.asarray(value)


def trace_tensor(stage: str, tensor: Any, **metadata: Any) -> None:
    """Append a deterministic tensor summary when tracing is enabled.

    Args:
        stage: Stable pipeline-stage identifier.
        tensor: PyTorch, MLX, NumPy, or array-like value to summarize.
        **metadata: JSON-serializable context such as backend, step, and timestep.
    """
    output_path = os.environ.get(TRACE_PATH_ENV)
    if not output_path:
        return

    import numpy as np

    array = _to_numpy(tensor)
    numeric = np.ascontiguousarray(array, dtype=np.float32)
    finite = numeric[np.isfinite(numeric)]
    record = {
        "stage": stage,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "count": int(numeric.size),
        "finite_count": int(finite.size),
        "sha256_f32": hashlib.sha256(numeric.tobytes()).hexdigest(),
        **metadata,
    }
    if finite.size:
        finite64 = finite.astype(np.float64)
        record.update(
            {
                "min": float(finite64.min()),
                "max": float(finite64.max()),
                "mean": float(finite64.mean()),
                "std": float(finite64.std()),
                "l2": float(np.linalg.norm(finite64)),
            }
        )

    path = Path(output_path).expanduser()
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
