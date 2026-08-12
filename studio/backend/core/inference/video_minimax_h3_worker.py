# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 external Torch/SageAttention worker bridge (experimental).

Patch 2 intentionally stops at a strict preflight contract. Studio can stay on its
pinned Python/Torch stack while an external interpreter (for example ``/usr/bin/python3``)
proves that the known-good H3 runtime can:

* import CUDA-enabled PyTorch + Triton + SageAttention,
* execute ``sageattn_qk_int8_pv_fp16_triton`` on the real GPU, and
* inspect the real pure-PEFT H3 Turbo adapter.

The worker communicates as one JSON request / one JSON response over stdio. No shell is
used. Actual H3 generation is deliberately NOT delegated in this patch; doing that only
after a base-weight/memory strategy is selected keeps a failed experiment from silently
falling back to the slow CPU-offload path we are trying to avoid.

Opt in with::

    UNSLOTH_H3_TORCH_WORKER_PYTHON=/usr/bin/python3

Optional::

    UNSLOTH_H3_TORCH_WORKER_TIMEOUT=90
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


H3_WORKER_PYTHON_ENV = "UNSLOTH_H3_TORCH_WORKER_PYTHON"
H3_WORKER_TIMEOUT_ENV = "UNSLOTH_H3_TORCH_WORKER_TIMEOUT"
DEFAULT_H3_WORKER_TIMEOUT = 90.0
_SAGE_KERNEL = "sageattn_qk_int8_pv_fp16_triton"
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"


def h3_worker_python_from_env() -> Optional[Path]:
    raw = (os.environ.get(H3_WORKER_PYTHON_ENV) or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{H3_WORKER_PYTHON_ENV} does not point to a file: {path}")
    if not os.access(path, os.X_OK):
        raise ValueError(f"{H3_WORKER_PYTHON_ENV} is not executable: {path}")
    return path


def _worker_timeout_from_env() -> float:
    raw = (os.environ.get(H3_WORKER_TIMEOUT_ENV) or str(DEFAULT_H3_WORKER_TIMEOUT)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{H3_WORKER_TIMEOUT_ENV} must be a number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{H3_WORKER_TIMEOUT_ENV} must be a finite positive number.")
    return value


def _last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("MiniMax-H3 worker returned no JSON object on stdout.")


def probe_h3_worker(
    python: Path,
    *,
    lora_path: Path,
    nfe: int,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Run the external runtime preflight and return its JSON capability record."""
    payload = {
        "action": "probe",
        "lora_path": str(Path(lora_path).expanduser().resolve()),
        "nfe": int(nfe),
    }
    command = [str(Path(python).expanduser().resolve()), str(Path(__file__).resolve()), "--worker"]
    try:
        result = subprocess.run(
            command,
            input = json.dumps(payload),
            capture_output = True,
            text = True,
            timeout = _worker_timeout_from_env() if timeout is None else float(timeout),
            check = False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"MiniMax-H3 Torch worker preflight timed out after {exc.timeout:g}s."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start MiniMax-H3 Torch worker: {exc}") from exc

    try:
        response = _last_json_object(result.stdout)
    except RuntimeError as exc:
        stderr = result.stderr.strip()
        if stderr:
            raise RuntimeError(f"{exc} Worker stderr: {stderr[-2000:]}") from exc
        raise

    if result.returncode != 0 or not response.get("ok"):
        error = str(response.get("error") or "worker exited unsuccessfully")
        stderr = result.stderr.strip()
        if stderr:
            error = f"{error}; stderr: {stderr[-2000:]}"
        raise RuntimeError(f"MiniMax-H3 Torch worker preflight failed: {error}")
    return response


def _inspect_lora(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("H3 worker requires the 'safetensors' package.") from exc

    if not path.is_file():
        raise ValueError(f"Turbo LoRA does not exist: {path}")

    with safe_open(path, framework = "pt", device = "cpu") as handle:
        keys = list(handle.keys())
        unsupported = [
            key
            for key in keys
            if not (key.endswith(_LORA_A_SUFFIX) or key.endswith(_LORA_B_SUFFIX))
        ]
        if unsupported:
            raise ValueError(
                "Turbo LoRA is not a pure PEFT checkpoint; unsupported keys include: "
                + ", ".join(unsupported[:3])
            )
        a_keys = [key for key in keys if key.endswith(_LORA_A_SUFFIX)]
        b_keys = {key for key in keys if key.endswith(_LORA_B_SUFFIX)}
        if not a_keys:
            raise ValueError("Turbo LoRA contains no PEFT A matrices.")
        first_a = a_keys[0]
        stem = first_a[: -len(_LORA_A_SUFFIX)]
        first_b = stem + _LORA_B_SUFFIX
        if first_b not in b_keys:
            raise ValueError(f"Turbo LoRA is missing the B matrix paired with {first_a}.")
        a_shape = tuple(handle.get_slice(first_a).get_shape())
        b_shape = tuple(handle.get_slice(first_b).get_shape())

    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
        raise ValueError(f"Turbo LoRA rank mismatch: A{a_shape}, B{b_shape}.")
    return {
        "path": str(path),
        "name": path.name,
        "tensor_count": len(keys),
        "rank": int(a_shape[0]),
    }


def _worker_probe(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("action") != "probe":
        raise ValueError("Unsupported H3 worker action.")
    nfe = int(payload.get("nfe", 0))
    if nfe < 1:
        raise ValueError("H3 Turbo NFE must be at least 1.")
    lora = _inspect_lora(Path(str(payload.get("lora_path", ""))).expanduser().resolve())

    import torch
    import triton
    import sageattention
    from sageattention import sageattn_qk_int8_pv_fp16_triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the H3 worker runtime.")

    # This is intentionally a REAL kernel invocation, not an import-only probe. It catches the
    # exact class of ABI/runtime mismatch that made Sage appear selectable but unusable earlier.
    q = torch.randn((1, 8, 512, 64), device = "cuda", dtype = torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    torch.cuda.synchronize()
    with torch.inference_mode():
        out = sageattn_qk_int8_pv_fp16_triton(
            q,
            k,
            v,
            tensor_layout = "HND",
            is_causal = False,
        )
    torch.cuda.synchronize()
    del q, k, v, out

    capability = torch.cuda.get_device_capability(0)
    return {
        "ok": True,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "sage_path": str(Path(sageattention.__file__).resolve()),
        "sage_kernel": _SAGE_KERNEL,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "lora": lora,
        "nfe": nfe,
    }


def _worker_main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("H3 worker request must be a JSON object.")
        response = _worker_probe(payload)
        print(json.dumps(response, separators = (",", ":")))
        return 0
    except Exception as exc:  # noqa: BLE001 -- process boundary must return structured failure
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                separators = (",", ":"),
            )
        )
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
    raise SystemExit("This module is an internal MiniMax-H3 worker. Use --worker.")
