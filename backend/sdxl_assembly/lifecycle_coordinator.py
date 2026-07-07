from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from backend.sdxl_assembly.progress import log_telemetry
from backend.sdxl_assembly.runtime_state import LifecycleDomain

logger = logging.getLogger(__name__)


class LifecycleChange(str, Enum):
    REQUEST_END = "request_end"
    MODEL_TYPE_CHANGE = "model_type_change"
    FAMILY_CHANGE = "family_change"
    SPINE_POSTURE_CHANGE = "spine_posture_change"
    MODEL_CHANGE = "model_change"
    CHECKPOINT_CHANGE = "checkpoint_change"
    LORA_STACK_CHANGE = "lora_stack_change"
    PROMPT_CHANGE = "prompt_change"
    SPATIAL_VAE_CHANGE = "spatial_vae_change"
    VAE_ARTIFACT_CHANGE = "vae_artifact_change"
    STRUCTURAL_CN_CHANGE = "structural_cn_change"
    CONTEXTUAL_CN_CHANGE = "contextual_cn_change"
    FULL_TEARDOWN = "full_teardown"


@dataclass(frozen=True)
class LifecycleReleasePlan:
    domains: tuple[LifecycleDomain, ...]
    reason: str
    changes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LifecycleReleaseError:
    domain: LifecycleDomain
    step: str
    error: Exception


@dataclass(frozen=True)
class LifecycleReleaseResult:
    plan: LifecycleReleasePlan
    errors: tuple[LifecycleReleaseError, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _normalize_domain(domain: Any) -> LifecycleDomain | None:
    if isinstance(domain, LifecycleDomain):
        return domain
    value = str(domain).strip().lower()
    for candidate in LifecycleDomain:
        if candidate.value == value or candidate.name.lower() == value:
            return candidate
    logger.warning("Unknown SDXL lifecycle domain: %s", domain)
    return None


def _normalize_domains(domain_or_domains: Any) -> tuple[LifecycleDomain, ...]:
    if domain_or_domains is None:
        return ()
    if isinstance(domain_or_domains, (str, LifecycleDomain)):
        raw_domains = (domain_or_domains,)
    else:
        raw_domains = tuple(domain_or_domains)

    normalized: list[LifecycleDomain] = []
    for raw in raw_domains:
        domain = _normalize_domain(raw)
        if domain is None:
            continue
        if domain == LifecycleDomain.FULL_TEARDOWN:
            for expanded in (
                LifecycleDomain.RUN_BOUND,
                LifecycleDomain.MODEL_PROMPT,
                LifecycleDomain.SPATIAL_VAE,
                LifecycleDomain.STRUCTURAL_CN,
                LifecycleDomain.CONTEXTUAL_CN,
            ):
                if expanded not in normalized:
                    normalized.append(expanded)
            continue
        if domain not in normalized:
            normalized.append(domain)
    return tuple(normalized)


def _normalize_change(change: Any) -> str:
    if isinstance(change, LifecycleChange):
        return change.value
    return str(change).strip().lower()


def plan_release_for_changes(
    changes: Iterable[Any],
    *,
    reason: str | None = None,
) -> LifecycleReleasePlan:
    normalized_changes = tuple(_normalize_change(change) for change in changes)
    domains: list[LifecycleDomain] = []

    def add(domain: LifecycleDomain) -> None:
        if domain == LifecycleDomain.MODEL_PROMPT and LifecycleDomain.PROMPT_CONDITIONING in domains:
            domains.remove(LifecycleDomain.PROMPT_CONDITIONING)
        if domain == LifecycleDomain.PROMPT_CONDITIONING and LifecycleDomain.MODEL_PROMPT in domains:
            return
        if domain not in domains:
            domains.append(domain)

    for change in normalized_changes:
        if change in {
            LifecycleChange.FAMILY_CHANGE.value,
            LifecycleChange.MODEL_TYPE_CHANGE.value,
            LifecycleChange.FULL_TEARDOWN.value,
        }:
            return LifecycleReleasePlan(
                domains=(LifecycleDomain.FULL_TEARDOWN,),
                reason=reason or change,
                changes=normalized_changes,
            )
        if change == LifecycleChange.REQUEST_END.value:
            add(LifecycleDomain.RUN_BOUND)
        elif change == LifecycleChange.PROMPT_CHANGE.value:
            add(LifecycleDomain.PROMPT_CONDITIONING)
        elif change in {
            LifecycleChange.SPINE_POSTURE_CHANGE.value,
            LifecycleChange.MODEL_CHANGE.value,
            LifecycleChange.CHECKPOINT_CHANGE.value,
            LifecycleChange.LORA_STACK_CHANGE.value,
        }:
            add(LifecycleDomain.MODEL_PROMPT)
        elif change in {
            LifecycleChange.SPATIAL_VAE_CHANGE.value,
            LifecycleChange.VAE_ARTIFACT_CHANGE.value,
        }:
            add(LifecycleDomain.SPATIAL_VAE)
        elif change == LifecycleChange.STRUCTURAL_CN_CHANGE.value:
            add(LifecycleDomain.STRUCTURAL_CN)
        elif change == LifecycleChange.CONTEXTUAL_CN_CHANGE.value:
            add(LifecycleDomain.CONTEXTUAL_CN)
        else:
            logger.warning("Unknown SDXL lifecycle change: %s", change)

    return LifecycleReleasePlan(
        domains=tuple(domains),
        reason=reason or "lifecycle_change",
        changes=normalized_changes,
    )


def _run_step(
    errors: list[LifecycleReleaseError],
    domain: LifecycleDomain,
    step_name: str,
    step: Callable[[], Any],
) -> None:
    try:
        step()
    except Exception as exc:
        logger.warning(
            "SDXL lifecycle release step failed domain=%s step=%s error=%s",
            domain.value,
            step_name,
            exc,
        )
        log_telemetry(
            "release_domain_step_error",
            f"domain={domain.value} step={step_name} error={exc.__class__.__name__}",
        )
        errors.append(LifecycleReleaseError(domain=domain, step=step_name, error=exc))


def _release_run_bound(errors: list[LifecycleReleaseError], assembly: Any | None) -> None:
    domain = LifecycleDomain.RUN_BOUND
    if assembly is None:
        log_telemetry("release_domain_step", f"domain={domain.value} step=no_active_assembly")
        return
    close = getattr(assembly, "close", None)
    if not callable(close):
        log_telemetry("release_domain_step", f"domain={domain.value} step=no_close_method")
        return
    _run_step(errors, domain, "assembly_close", close)


def _release_model_prompt(errors: list[LifecycleReleaseError], reason: str) -> None:
    from backend.sdxl_assembly import runtime_state

    domain = LifecycleDomain.MODEL_PROMPT
    _run_step(
        errors,
        domain,
        "streaming_spine",
        lambda: runtime_state.release_active_sdxl_streaming_spine(reason=reason),
    )
    _run_step(
        errors,
        domain,
        "text_encoder_cache",
        lambda: runtime_state.release_text_encoder_component_cache(reason=reason),
    )
    _run_step(
        errors,
        domain,
        "prompt_conditioning_cache",
        lambda: runtime_state.release_prompt_conditioning_cache(reason=reason),
    )

    def clear_lora_cache() -> None:
        from backend.sdxl_assembly.cpu_lora_worker import _PARSED_LORA_CACHE

        _PARSED_LORA_CACHE.clear()

    _run_step(errors, domain, "parsed_lora_cache", clear_lora_cache)


def _release_prompt_conditioning(errors: list[LifecycleReleaseError], reason: str) -> None:
    from backend.sdxl_assembly import runtime_state

    domain = LifecycleDomain.PROMPT_CONDITIONING
    _run_step(
        errors,
        domain,
        "prompt_conditioning_cache",
        lambda: runtime_state.release_prompt_conditioning_cache(reason=reason),
    )


def _release_spatial_vae(errors: list[LifecycleReleaseError]) -> None:
    def clear_encode_cache() -> None:
        from backend.sdxl_assembly.vae_encode_worker import VaeEncodeWorker

        VaeEncodeWorker._ENCODE_CACHE.clear()

    _run_step(errors, LifecycleDomain.SPATIAL_VAE, "vae_encode_cache", clear_encode_cache)


def _release_structural_cn(errors: list[LifecycleReleaseError]) -> None:
    def clear_preprocess_cache() -> None:
        from backend.sdxl_assembly.stream_st_preprocess_worker import StreamingStructuralPreprocessWorker

        StreamingStructuralPreprocessWorker.clear_preprocess_cache()

    def clear_support_cache() -> None:
        from backend.sdxl_assembly.stream_st_cn_worker import StreamingStructuralControlWorker

        StreamingStructuralControlWorker.clear_support_cache()

    domain = LifecycleDomain.STRUCTURAL_CN
    _run_step(errors, domain, "structural_preprocess_cache", clear_preprocess_cache)
    _run_step(errors, domain, "structural_support_cache", clear_support_cache)


def _release_contextual_cn(errors: list[LifecycleReleaseError]) -> None:
    def clear_payload_cache() -> None:
        from backend.sdxl_assembly.stream_ctx_cn_worker import StreamingContextualControlWorker

        StreamingContextualControlWorker.clear_payload_cache()

    def clear_support_cache() -> None:
        from backend.sdxl_assembly.stream_ctx_cn_worker import StreamingContextualControlWorker

        StreamingContextualControlWorker.clear_support_cache()

    domain = LifecycleDomain.CONTEXTUAL_CN
    _run_step(errors, domain, "contextual_payload_cache", clear_payload_cache)
    _run_step(errors, domain, "contextual_support_cache", clear_support_cache)


def release_domains(
    domain_or_domains: Any,
    *,
    reason: str | None = None,
    assembly: Any | None = None,
    raise_on_error: bool = False,
) -> LifecycleReleaseResult:
    is_full_teardown = False
    if domain_or_domains == LifecycleDomain.FULL_TEARDOWN:
        is_full_teardown = True
    elif isinstance(domain_or_domains, (list, tuple, set)):
        if LifecycleDomain.FULL_TEARDOWN in domain_or_domains:
            is_full_teardown = True

    if is_full_teardown:
        try:
            from backend.sdxl_assembly.gateway import clear_gateway_state
            clear_gateway_state()
        except Exception:
            pass

    clear_reason = reason or "unspecified"
    plan = LifecycleReleasePlan(
        domains=_normalize_domains(domain_or_domains),
        reason=clear_reason,
    )
    errors: list[LifecycleReleaseError] = []

    for domain in plan.domains:
        logger.debug("[SDXL Telemetry] Releasing domain %s, reason=%s", domain.value, clear_reason)
        log_telemetry("release_domain", f"domain={domain.value} reason={clear_reason}")
        if domain == LifecycleDomain.RUN_BOUND:
            _release_run_bound(errors, assembly)
        elif domain == LifecycleDomain.PROMPT_CONDITIONING:
            _release_prompt_conditioning(errors, clear_reason)
        elif domain == LifecycleDomain.MODEL_PROMPT:
            _release_model_prompt(errors, clear_reason)
        elif domain == LifecycleDomain.SPATIAL_VAE:
            _release_spatial_vae(errors)
        elif domain == LifecycleDomain.STRUCTURAL_CN:
            _release_structural_cn(errors)
        elif domain == LifecycleDomain.CONTEXTUAL_CN:
            _release_contextual_cn(errors)

    gc.collect()
    result = LifecycleReleaseResult(plan=plan, errors=tuple(errors))
    if raise_on_error and errors:
        first = errors[0]
        raise RuntimeError(
            f"SDXL lifecycle release failed domain={first.domain.value} step={first.step}"
        ) from first.error
    return result


def release_for_changes(
    changes: Iterable[Any],
    *,
    reason: str | None = None,
    assembly: Any | None = None,
    raise_on_error: bool = False,
) -> LifecycleReleaseResult:
    plan = plan_release_for_changes(changes, reason=reason)
    return release_domains(
        plan.domains,
        reason=plan.reason,
        assembly=assembly,
        raise_on_error=raise_on_error,
    )
