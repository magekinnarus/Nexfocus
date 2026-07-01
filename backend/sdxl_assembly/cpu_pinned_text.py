from __future__ import annotations

import time
import logging
from typing import Any, Dict
from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry
from backend.sdxl_assembly.runtime_state import (
    acquire_text_encoder_component,
    lookup_prompt_conditioning,
    remember_prompt_conditioning,
    release_text_encoder_component_cache,
)
from backend.sdxl_assembly.streaming_lora import StreamingLoraPatchWorker

logger = logging.getLogger(__name__)

class CpuPinnedTextEncoderWorker:
    """Worker representing CpuPinnedTextEncoderWorker (CLIP-L and CLIP-G on CPU)."""
    
    def __init__(self, request: SDXLAssemblyRequest, lora_worker: StreamingLoraPatchWorker | None = None) -> None:
        self.request = request
        self.lora_worker = lora_worker or StreamingLoraPatchWorker(request)

    def get_conditioning(self) -> Dict[str, Any]:
        """Encodes positive and negative prompts on CPU.
        
        Uses isolated conditioning cache to reuse prompt conditioning fast.
        """
        log_telemetry("prompt_encode_begin")
        
        # 1. Check prompt cache first
        cached_cond = lookup_prompt_conditioning(self.request)
        if cached_cond is not None:
            log_telemetry("prompt_encode_complete", "cache_hit=True")
            return cached_cond

        # 2. Cache MISS: Load owned CLIP, apply LoRAs, and encode
        start_time = time.perf_counter()
        clip = None
        try:
            clip = acquire_text_encoder_component(self.request)
            
            # Apply LoRAs to CLIP patcher if clip_weight != 0
            self.lora_worker.apply_clip_patches(clip)
            
            if self.lora_worker.clip_patch_count > 0:
                from backend.cpu_compiler import CpuArtifactCompiler
                logger.debug("[SDXL Telemetry] Compiling %d LoRA patches onto CLIP on CPU...", self.lora_worker.clip_patch_count)
                CpuArtifactCompiler.compile_patcher(clip.patcher)
                
            from backend import conditioning
            
            # Respect clip layer skip
            if hasattr(clip, "clip_layer"):
                clip.clip_layer(self.request.clip_layer)
                
            encoded_prompt_pair = conditioning.encode_prompt_pair_sdxl(
                clip,
                self.request.prompt,
                self.request.negative_prompt,
                positive_texts=self.request.positive_texts,
                negative_texts=self.request.negative_texts,
                use_explicit_residency=True,
            )
            
            # Build ADM Scale Pair
            adm_pair = conditioning.build_sdxl_adm_pair(
                encoded_prompt_pair,
                self.request.width,
                self.request.height,
                target_width=self.request.width,
                target_height=self.request.height,
                adm_scale_positive=self.request.adm_scaler_positive,
                adm_scale_negative=self.request.adm_scaler_negative,
            )
            
            # Format for sampler
            positive = [[
                encoded_prompt_pair["positive"]["cond"],
                {
                    "pooled_output": encoded_prompt_pair["positive"]["pooled"],
                    "model_conds": {"y": adm_pair["positive"]},
                },
            ]]
            negative = [[
                encoded_prompt_pair["negative"]["cond"],
                {
                    "pooled_output": encoded_prompt_pair["negative"]["pooled"],
                    "model_conds": {"y": adm_pair["negative"]},
                },
            ]]
            
            conditioning_payload = {
                "positive": positive,
                "negative": negative,
            }
            
            remember_prompt_conditioning(self.request, conditioning_payload)
            
            duration = time.perf_counter() - start_time
            log_telemetry("prompt_encode_complete", f"cache_hit=False duration={duration:.3f}s")
            return conditioning_payload
        finally:
            clip = None
            import gc
            gc.collect()

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        if bool(self.request.metadata.get("release_text_encoder_after_task", False)):
            release_text_encoder_component_cache(reason="assembly_close")
