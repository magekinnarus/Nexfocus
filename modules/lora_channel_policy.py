from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional

from backend.cpu_compiler import SafeOpenHeaderOnly

logger = logging.getLogger(__name__)

PRESET_SPEED_LORAS = frozenset(
    {
        "sdxl_lcm_lora.safetensors",
        "sdxl_lightning_4step_lora.safetensors",
        "sdxl_lightning_8step_lora.safetensors",
    }
)


@dataclass(frozen=True)
class LoRAAssetEvidence:
    sha256: str
    size_bytes: int
    modified_ns: int
    status: str  # "recognized", "unknown", "error", "skipped"
    unet_count: int
    clip_l_count: int
    clip_g_count: int
    generic_text_count: int
    recognized_key_count: int
    unrecognized_key_count: int


@dataclass(frozen=True)
class LoRAChannelDecision:
    requested_unet_weight: float
    requested_clip_weight: float
    effective_unet_weight: float
    effective_clip_weight: float
    source: str  # "explicit", "asset_evidence", "conservative_default"
    evidence_status: str  # "recognized", "unknown", "error", "skipped", "unavailable"
    reason: str


# Bounded file-identity cache (LRU cache of size 100)
_EVIDENCE_CACHE: OrderedDict[Tuple[str, int, int], LoRAAssetEvidence] = OrderedDict()
_EVIDENCE_CACHE_LIMIT = 100


def _is_numbered_text_encoder_key(lower_key: str) -> bool:
    """Recognize generic ``lora_teN_`` keys beyond SDXL's CLIP-L/CLIP-G."""
    prefix = "lora_te"
    if not lower_key.startswith(prefix):
        return False
    remainder = lower_key[len(prefix):]
    digits = remainder[: len(remainder) - len(remainder.lstrip("0123456789"))]
    return bool(digits) and remainder[len(digits):].startswith("_") and int(digits) >= 3


def inspect_lora_asset(file_identity: Any) -> LoRAAssetEvidence:
    """Inspects a LoRA asset's safetensors header to classify target channels."""
    sha256 = getattr(file_identity, "sha256", None) or ""
    size_bytes = int(getattr(file_identity, "size_bytes", 0) or 0)
    modified_ns = int(getattr(file_identity, "modified_ns", 0) or 0)
    path = getattr(file_identity, "path", None)
    path_str = str(path) if path else ""

    cache_key = (sha256, size_bytes, modified_ns)
    if sha256 and cache_key in _EVIDENCE_CACHE:
        evidence = _EVIDENCE_CACHE[cache_key]
        _EVIDENCE_CACHE.move_to_end(cache_key)
        return evidence

    # Treat non-safetensors conservatively as unknown
    lower_path = path_str.lower()
    if not lower_path.endswith(".safetensors"):
        evidence = LoRAAssetEvidence(
            sha256=sha256,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            status="unknown",
            unet_count=0,
            clip_l_count=0,
            clip_g_count=0,
            generic_text_count=0,
            recognized_key_count=0,
            unrecognized_key_count=0,
        )
        if sha256:
            _EVIDENCE_CACHE[cache_key] = evidence
            if len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
        return evidence

    try:
        header = SafeOpenHeaderOnly(path_str)
        unet_count = 0
        clip_l_count = 0
        clip_g_count = 0
        generic_text_count = 0
        unrecognized_count = 0

        # Standard LoRA parameter suffixes
        lora_suffixes = [
            ".lora_down.weight", ".lora_up.weight", ".lora_down.bias", ".lora_up.bias",
            ".alpha", ".dora_scale", ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b",
            ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2", ".hada_w1_a", ".hada_w1_b",
            ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2", ".a1.weight",
            ".a2.weight", ".b1.weight", ".b2.weight", ".w_norm", ".b_norm",
            ".diff", ".diff_b", ".set_weight"
        ]

        for key in header.keys():
            lower_key = key.lower()
            is_lora_param = any(suffix in lower_key for suffix in lora_suffixes)
            if not is_lora_param:
                continue

            # Classify keys
            is_unet = (
                "lora_unet_" in lower_key or
                lower_key.startswith("diffusion_model.") or
                lower_key.startswith(("down_blocks.", "up_blocks.", "mid_block.")) or
                lower_key.startswith("unet.") or
                "lycoris_" in lower_key
            )
            is_clip_l = (
                "lora_te1_" in lower_key or
                "lora_te_" in lower_key or
                lower_key.startswith("text_encoder.text_model.encoder.layers.") or
                lower_key.startswith("text_encoder.transformer.text_model.encoder.layers.") or
                lower_key.startswith("text_encoder.transformer.encoder.layers.")
            )
            is_clip_g = (
                "lora_te2_" in lower_key or
                lower_key.startswith("text_encoder_2.text_model.encoder.layers.") or
                lower_key.startswith("text_encoder_2.transformer.text_model.encoder.layers.") or
                lower_key.startswith("text_encoder_2.transformer.encoder.layers.")
            )
            is_generic_text = (
                lower_key.startswith("text_encoders.") or
                lower_key.startswith("lora_prior_te_") or
                "t5xxl.transformer." in lower_key or
                "hydit_clip.transformer.bert." in lower_key or
                _is_numbered_text_encoder_key(lower_key)
            )

            if is_unet:
                unet_count += 1
            elif is_clip_l:
                clip_l_count += 1
            elif is_clip_g:
                clip_g_count += 1
            elif is_generic_text:
                generic_text_count += 1
            else:
                unrecognized_count += 1

        recognized_count = unet_count + clip_l_count + clip_g_count + generic_text_count
        # A header containing only unsupported conventions is still unknown.
        # It must retain conservative requested channels and must not be
        # reported as recognized merely because it contained LoRA-like keys.
        status = "recognized" if recognized_count > 0 else "unknown"

        evidence = LoRAAssetEvidence(
            sha256=sha256,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            status=status,
            unet_count=unet_count,
            clip_l_count=clip_l_count,
            clip_g_count=clip_g_count,
            generic_text_count=generic_text_count,
            recognized_key_count=recognized_count,
            unrecognized_key_count=unrecognized_count,
        )
    except Exception as e:
        logger.error("Failed to inspect LoRA asset %s: %s", path_str, str(e))
        evidence = LoRAAssetEvidence(
            sha256=sha256,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            status="error",
            unet_count=0,
            clip_l_count=0,
            clip_g_count=0,
            generic_text_count=0,
            recognized_key_count=0,
            unrecognized_key_count=0,
        )

    if sha256:
        _EVIDENCE_CACHE[cache_key] = evidence
        if len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
            _EVIDENCE_CACHE.popitem(last=False)
    return evidence


def resolve_lora_channels(
    file_identity: Optional[Any],
    requested_unet_weight: float,
    requested_clip_weight: float,
    provenance: str,
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    raw_path: Optional[str] = None,
) -> LoRAChannelDecision:
    """Decides the effective channel weights based on explicit overrides, presets, and asset evidence."""
    path = getattr(file_identity, "path", None) if file_identity else None
    path_str = str(path) if path else (raw_path or "")
    basename = get_lora_basename(path_str) if path_str else ""

    # 1. Explicit inpaint / additional provenance contract
    if provenance == "additional":
        return LoRAChannelDecision(
            requested_unet_weight=requested_unet_weight,
            requested_clip_weight=requested_clip_weight,
            effective_unet_weight=requested_unet_weight,
            effective_clip_weight=0.0,
            source="explicit",
            evidence_status="skipped",
            reason="Additional base-model LoRAs are explicitly UNet-only.",
        )

    # 2. Explicit override dictionary / preset-managed contract
    if overrides and path_str:
        override = resolve_lora_channel_override(overrides, path_str)
        if isinstance(override, dict):
            target = str(override.get("target", "")).strip().lower()
            if target == "unet_only":
                eff_unet = requested_unet_weight
                eff_clip = float(override.get("clip_weight", 0.0) or 0.0)
                source = "explicit"
                if basename and is_preset_speed_lora(basename):
                    reason = f"Preset speed LoRA '{basename}' is explicitly UNet-only via preset overrides."
                else:
                    reason = f"Explicit override applied: target={target}, clip_weight={eff_clip}."
                return LoRAChannelDecision(
                    requested_unet_weight=requested_unet_weight,
                    requested_clip_weight=requested_clip_weight,
                    effective_unet_weight=eff_unet,
                    effective_clip_weight=eff_clip,
                    source=source,
                    evidence_status="skipped",
                    reason=reason,
                )

    # 3. Recognized asset evidence (if file is local and exists)
    if path_str and os.path.exists(path_str):
        evidence = inspect_lora_asset(file_identity)
        if evidence.status == "recognized":
            has_unet = evidence.unet_count > 0
            has_text = (evidence.clip_l_count > 0 or evidence.clip_g_count > 0 or evidence.generic_text_count > 0)

            if has_unet and has_text:
                return LoRAChannelDecision(
                    requested_unet_weight=requested_unet_weight,
                    requested_clip_weight=requested_clip_weight,
                    effective_unet_weight=requested_unet_weight,
                    effective_clip_weight=requested_clip_weight,
                    source="asset_evidence",
                    evidence_status="recognized",
                    reason=f"Asset contains {evidence.unet_count} UNet keys and {evidence.clip_l_count + evidence.clip_g_count + evidence.generic_text_count} text keys. Effective: dual-target.",
                )
            elif has_unet and not has_text:
                return LoRAChannelDecision(
                    requested_unet_weight=requested_unet_weight,
                    requested_clip_weight=requested_clip_weight,
                    effective_unet_weight=requested_unet_weight,
                    effective_clip_weight=0.0,
                    source="asset_evidence",
                    evidence_status="recognized",
                    reason=f"Asset contains {evidence.unet_count} UNet keys and 0 text keys. Effective: UNet-only.",
                )
            elif not has_unet and has_text:
                return LoRAChannelDecision(
                    requested_unet_weight=requested_unet_weight,
                    requested_clip_weight=requested_clip_weight,
                    effective_unet_weight=0.0,
                    effective_clip_weight=requested_clip_weight,
                    source="asset_evidence",
                    evidence_status="recognized",
                    reason=f"Asset contains 0 UNet keys and {evidence.clip_l_count + evidence.clip_g_count + evidence.generic_text_count} text keys. Effective: text-only.",
                )
            else:
                pass
        evidence_status = evidence.status
    else:
        evidence_status = "unavailable"

    # 4. Unknown or unavailable evidence: conservative fallback
    return LoRAChannelDecision(
        requested_unet_weight=requested_unet_weight,
        requested_clip_weight=requested_clip_weight,
        effective_unet_weight=requested_unet_weight,
        effective_clip_weight=requested_clip_weight,
        source="conservative_default",
        evidence_status=evidence_status,
        reason=f"Conservative default fallback: evidence status is '{evidence_status}'.",
    )


def normalize_lora_override_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def get_lora_basename(value: Any) -> str:
    normalized = normalize_lora_override_key(value)
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1]


def is_preset_speed_lora(value: Any) -> bool:
    return get_lora_basename(value) in PRESET_SPEED_LORAS


def iter_lora_override_keys(value: Any) -> tuple[str, ...]:
    normalized = normalize_lora_override_key(value)
    if not normalized:
        return ()
    basename = get_lora_basename(normalized)
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
        if not is_preset_speed_lora(lora_path):
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
