# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Experimental MiniMax-H3 Turbo LoRA support for the Diffusers video path.

This is deliberately H3-only. The first integration target is a *pure PEFT* LoRA
checkpoint such as the 4-NFE preview released by ModelTC/LightX2V. ComfyUI-specific
Turbo checkpoints can carry additional time-conditioning behaviour for pruned bases;
those need a separate compatibility layer and are intentionally rejected here rather
than being applied incorrectly.

The feature is opt-in through environment variables for now so it can be exercised
without changing the public Studio API while the backend contract is validated:

  UNSLOTH_H3_TURBO_LORA=/absolute/path/to/adapter.safetensors
  UNSLOTH_H3_TURBO_STEPS=4
  UNSLOTH_H3_TURBO_ALPHA=8
  UNSLOTH_H3_TURBO_SCALE=1.0
  UNSLOTH_H3_TURBO_FUSE=0
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


H3_TURBO_LORA_ENV = "UNSLOTH_H3_TURBO_LORA"
H3_TURBO_STEPS_ENV = "UNSLOTH_H3_TURBO_STEPS"
H3_TURBO_ALPHA_ENV = "UNSLOTH_H3_TURBO_ALPHA"
H3_TURBO_SCALE_ENV = "UNSLOTH_H3_TURBO_SCALE"
H3_TURBO_FUSE_ENV = "UNSLOTH_H3_TURBO_FUSE"

DEFAULT_H3_TURBO_NFE = 4
DEFAULT_H3_TURBO_ALPHA = 8

LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"


@dataclass(frozen=True)
class H3TurboConfig:
    path: Path
    nfe: int = DEFAULT_H3_TURBO_NFE
    alpha: int = DEFAULT_H3_TURBO_ALPHA
    scale: float = 1.0
    fuse: bool = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be one of 0/1, false/true, no/yes, off/on.")


def h3_turbo_config_from_env() -> H3TurboConfig | None:
    """Return the opt-in Turbo configuration, or ``None`` when it is disabled."""
    raw_path = (os.environ.get(H3_TURBO_LORA_ENV) or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{H3_TURBO_LORA_ENV} does not point to a file: {path}")

    try:
        nfe = int((os.environ.get(H3_TURBO_STEPS_ENV) or str(DEFAULT_H3_TURBO_NFE)).strip())
    except ValueError as exc:
        raise ValueError(f"{H3_TURBO_STEPS_ENV} must be an integer.") from exc
    if nfe < 1:
        raise ValueError(f"{H3_TURBO_STEPS_ENV} must be at least 1.")

    try:
        alpha = int((os.environ.get(H3_TURBO_ALPHA_ENV) or str(DEFAULT_H3_TURBO_ALPHA)).strip())
    except ValueError as exc:
        raise ValueError(f"{H3_TURBO_ALPHA_ENV} must be an integer.") from exc
    if alpha < 1:
        raise ValueError(f"{H3_TURBO_ALPHA_ENV} must be at least 1.")

    try:
        scale = float((os.environ.get(H3_TURBO_SCALE_ENV) or "1.0").strip())
    except ValueError as exc:
        raise ValueError(f"{H3_TURBO_SCALE_ENV} must be a number.") from exc
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"{H3_TURBO_SCALE_ENV} must be a finite non-negative number.")

    return H3TurboConfig(
        path=path,
        nfe=nfe,
        alpha=alpha,
        scale=scale,
        fuse=_env_bool(H3_TURBO_FUSE_ENV),
    )


def h3_turbo_scheduler_grid_points(nfe: int) -> int:
    """Translate user-visible H3 transformer evaluations to Diffusers sigma grid points.

    MiniMaxH3Scheduler includes terminal sigma=0 in ``num_inference_steps``. Therefore N
    transformer evaluations require N+1 grid points for the distilled Turbo schedule.
    """
    nfe = int(nfe)
    if nfe < 1:
        raise ValueError("MiniMax-H3 Turbo steps must be at least 1.")
    return nfe + 1


def _load_lora_state_dict(path: Path) -> Mapping[str, Any]:
    import torch

    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MiniMax-H3 Turbo .safetensors loading requires the 'safetensors' package."
            ) from exc
        checkpoint = load_safetensors_file(str(path), device="cpu")
    else:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("state_dict"), Mapping):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a state-dict mapping in MiniMax-H3 Turbo LoRA {path}, "
            f"got {type(checkpoint).__name__}."
        )
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError(f"MiniMax-H3 Turbo LoRA contains non-string keys: {path}")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError(f"MiniMax-H3 Turbo LoRA contains non-tensor values: {path}")
    return checkpoint


def _validate_pure_peft_lora(state_dict: Mapping[str, Any], path: Path) -> int:
    lora_a: dict[str, Any] = {}
    lora_b: dict[str, Any] = {}
    unsupported: list[str] = []
    for key, tensor in state_dict.items():
        if key.endswith(_LORA_A_SUFFIX):
            lora_a[key[: -len(_LORA_A_SUFFIX)]] = tensor
        elif key.endswith(_LORA_B_SUFFIX):
            lora_b[key[: -len(_LORA_B_SUFFIX)]] = tensor
        else:
            unsupported.append(key)

    if unsupported:
        preview = ", ".join(unsupported[:3])
        raise ValueError(
            f"{path} is not a pure PEFT MiniMax-H3 Turbo LoRA state dict; "
            f"unsupported keys include: {preview}. ComfyUI-specific H3 Turbo checkpoints "
            "with extra time-conditioning need a separate compatibility path."
        )
    if not lora_a:
        raise ValueError(f"No {_LORA_A_SUFFIX} tensors found in MiniMax-H3 Turbo LoRA {path}.")

    missing_a = sorted(lora_b.keys() - lora_a.keys())
    missing_b = sorted(lora_a.keys() - lora_b.keys())
    if missing_a or missing_b:
        raise ValueError(
            "Unpaired MiniMax-H3 Turbo LoRA tensors: "
            f"missing A={missing_a[:3]}, missing B={missing_b[:3]}."
        )

    ranks: set[int] = set()
    for module_name in lora_a:
        a_tensor = lora_a[module_name]
        b_tensor = lora_b[module_name]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {module_name} must be matrices, got "
                f"A{tuple(a_tensor.shape)} and B{tuple(b_tensor.shape)}."
            )
        if a_tensor.shape[0] != b_tensor.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: "
                f"A{tuple(a_tensor.shape)} and B{tuple(b_tensor.shape)}."
            )
        if not module_name.endswith(LORA_TARGET_MODULES):
            raise ValueError(f"Unsupported MiniMax-H3 Turbo LoRA target module {module_name!r}.")
        ranks.add(int(a_tensor.shape[0]))
    if len(ranks) != 1:
        raise ValueError(f"Mixed MiniMax-H3 Turbo LoRA ranks are unsupported: {sorted(ranks)}.")
    return ranks.pop()


def apply_h3_turbo_lora(transformer: Any, config: H3TurboConfig, *, logger: Any = None) -> int:
    """Inject a pure-PEFT Turbo LoRA into the already-loaded H3 denoiser.

    Returns the detected LoRA rank. The adapter remains separate by default because that is the
    safer path for quantized base weights. Fusion is an explicit opt-in.
    """
    try:
        from peft import LoraConfig
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MiniMax-H3 Turbo LoRA support requires PEFT. Install/enable the Studio Diffusers "
            "dependencies that provide the 'peft' package."
        ) from exc

    state_dict = _load_lora_state_dict(config.path)
    rank = _validate_pure_peft_lora(state_dict, config.path)
    if not hasattr(transformer, "add_adapter"):
        raise RuntimeError(
            "The loaded MiniMax-H3 transformer does not expose Diffusers/PEFT adapter hooks."
        )

    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=config.alpha,
            init_lora_weights=False,
            target_modules=list(LORA_TARGET_MODULES),
            use_rslora=False,
        )
    )
    adapter_parameters = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    missing = sorted(adapter_parameters.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - adapter_parameters.keys())
    shape_mismatches = [
        (name, tuple(state_dict[name].shape), tuple(parameter.shape))
        for name, parameter in adapter_parameters.items()
        if name in state_dict and state_dict[name].shape != parameter.shape
    ]
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "MiniMax-H3 Turbo LoRA is incompatible with the loaded transformer: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"shape_mismatches={shape_mismatches[:3]}."
        )

    incompatible = transformer.load_state_dict(state_dict, strict=False)
    missing_lora = [
        key
        for key in incompatible.missing_keys
        if ".lora_A." in key or ".lora_B." in key
    ]
    if incompatible.unexpected_keys or missing_lora:
        raise RuntimeError(
            "MiniMax-H3 Turbo LoRA loading did not consume the expected adapter tensors: "
            f"missing={missing_lora[:3]}, unexpected={incompatible.unexpected_keys[:3]}."
        )

    transformer.set_adapters("default", weights=config.scale)
    if config.fuse:
        if not hasattr(transformer, "fuse_lora") or not hasattr(transformer, "unload_lora"):
            raise RuntimeError("The loaded MiniMax-H3 transformer cannot fuse/unload PEFT LoRAs.")
        transformer.fuse_lora(lora_scale=1.0, safe_fusing=True, adapter_names=["default"])
        transformer.unload_lora()
    transformer.requires_grad_(False)
    transformer.eval()
    if logger is not None:
        logger.info(
            "video.h3_turbo_lora: loaded %s rank=%d alpha=%d scale=%g fused=%s",
            config.path,
            rank,
            config.alpha,
            config.scale,
            config.fuse,
        )
    return rank
