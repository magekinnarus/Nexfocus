from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


PRESET_SPEED_LORAS = frozenset(
    {
        "sdxl_lcm_lora.safetensors",
        "sdxl_lightning_4step_lora.safetensors",
        "sdxl_lightning_8step_lora.safetensors",
    }
)


def normalize_lora_override_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def iter_lora_override_keys(value: Any) -> tuple[str, ...]:
    normalized = normalize_lora_override_key(value)
    if not normalized:
        return ()
    basename = normalized.rsplit("/", 1)[-1]
    if basename == normalized:
        return (normalized,)
    return (normalized, basename)


def build_explicit_lora_channel_overrides(
    loras: Iterable[tuple[Any, Any]],
) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    for lora_path, _weight in loras or ():
        keys = iter_lora_override_keys(lora_path)
        if not keys:
            continue
        if keys[-1] not in PRESET_SPEED_LORAS:
            continue
        override = {
            "target": "unet_only",
            "clip_weight": 0.0,
            "source": "preset_speed_lora",
        }
        for key in keys:
            overrides[key] = dict(override)
    return overrides


def merge_lora_channel_overrides(
    *overrides_maps: Mapping[str, Mapping[str, Any]] | None,
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for overrides in overrides_maps:
        if not isinstance(overrides, Mapping):
            continue
        for raw_key, raw_value in overrides.items():
            keys = iter_lora_override_keys(raw_key)
            if not keys or not isinstance(raw_value, Mapping):
                continue
            normalized_value = dict(raw_value)
            for key in keys:
                merged[key] = dict(normalized_value)
    return merged


def resolve_lora_channel_override(
    overrides: Mapping[str, Mapping[str, Any]] | None,
    lora_path: Any,
) -> Dict[str, Any] | None:
    if not isinstance(overrides, Mapping):
        return None
    for key in iter_lora_override_keys(lora_path):
        value = overrides.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return None
