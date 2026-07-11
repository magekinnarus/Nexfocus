# Runtime Validation

This file defines the canonical automated validation contract for the post-W11
runtime model.

Historical mission work reports still describe what was true when each work
order closed. Use the commands in this file for current validation and closure
evidence.

## Environment

Run validation through the project virtual environment:

```powershell
.\venv\Scripts\python.exe tools\check_validation_env.py
```

Validated local W12 baseline:

- `transformers==4.44.2`
- `huggingface-hub==0.36.2`
- `tokenizers==0.19.1`
- `accelerate==1.13.0`

Known mismatch:

- Plain system Python on this machine currently carries an incompatible
  `huggingface-hub==1.4.1`, which breaks `transformers` import and therefore
  must not be used for mission validation.

## Optional Launch Overrides

For constrained-hardware reproduction on a roomier Colab Pro session, the UI
launch path now supports:

- `--memory-environment-profile`
- `--hardware-total-ram-mb`
- `--hardware-total-vram-mb`

Example constrained Colab streaming simulation:

- keep `--colab`
- keep the roomy Colab Pro RAM budget
- add `--hardware-total-vram-mb 12288`
- if you want the simulation fully explicit, also add `--hardware-total-ram-mb 57344`
- leave `--memory-environment-profile` on `auto` or pin it to `colab_pro`
- do not use `colab_free` for fp8 Flux Fill streaming validation; that profile exists because free-tier RAM is too small for the target streaming test

Resident SDXL no longer exposes a clean-shadow placement override. The live
runtime always reloads clean UNet weights from the authoritative runtime source
instead of keeping a separate CPU/GPU clean snapshot policy.

Flux Fill runtime posture now exposes only:

- `auto`
- `streaming`

Use `auto` for ordinary validation. Use `streaming` only as a benchmark/debug
override on roomy hardware such as Colab Pro when you explicitly want to force
the streaming lane. There is no separate `resident` UI override because resident
machines should naturally resolve to `auto`, while streaming-capable benchmark
machines may opt into streaming explicitly.

## Search And Compile

Ownership/runtime audit search:

```powershell
rg -n "sdxl_runtime_owner|process_diffusion|runtime_family|execution_mode|gguf_sdxl|flux_fill" backend modules tools tracked_tests tests
```

Compile sanity on the authoritative runtime surfaces:

```powershell
$fluxV3Files = Get-ChildItem backend\flux_fill_v3\*.py | ForEach-Object { $_.FullName }
.\venv\Scripts\python.exe -m py_compile @fluxV3Files backend\memory_governor.py backend\resources.py backend\sdxl_runtime_policy.py backend\sdxl_streaming_runtime.py backend\sdxl_unified_runtime.py backend\staging_manager.py backend\sdxl_assembly\cpu_text_encode_worker.py backend\sdxl_assembly\progress.py backend\sdxl_assembly\runtime_state.py modules\async_worker.py modules\objr_engine.py modules\parameter_registry.py modules\pipeline\inference.py modules\pipeline\routes.py modules\pipeline\tiled_refinement.py modules\runtime_surface_state.py modules\runtime_surface_api.py modules\task_state.py modules\ui_components\advanced_panel.py modules\ui_logic.py webui.py tools\check_validation_env.py tracked_tests\test_memory_residency.py tracked_tests\test_pipeline_routes.py tracked_tests\test_pipeline_stage_runtime.py tracked_tests\test_sdxl_assembly_w03_regression.py tracked_tests\test_sdxl_assembly_w04_regression.py tracked_tests\test_sdxl_assembly_w10b_lifecycle_coordinator.py tests\test_runtime_surface_api.py tests\test_sdxl_assembly_w10d.py tests\test_sdxl_outer_wiring_w10c.py
```

## Regression Matrix

### 1. Unified SDXL Runtime And Image-Input Handoff

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_unified_runtime.py tests\test_unified_runtime_handoff.py tests\test_gguf_runtime_handoff.py tests\test_async_worker_process_transition.py tests\test_default_pipeline_process_reset.py tests\test_super_upscale_residency.py -q
```

Covers:

- standard unified SDXL route
- unified SDXL image-input route
- authoritative runtime handoff
- GGUF compatibility/quarantine expectations
- process-transition and cleanup behavior
- tiled-refinement runtime dispatch and interrupt semantics

### 2. Authoritative Pipeline Consolidation And GGUF Seam

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_gguf_dispatch_seam.py -q
```

Covers:

- retained GGUF dispatch seam classification
- explicit compatibility-lane expectations

### 3. Runtime-Centered Memory / Hardware / Flux Fill / Runtime-Surface Sanity

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_memory_governor.py tests\test_w11_policy_simplification.py tests\test_async_worker_process_transition.py tests\test_flux_fill_v3.py tests\test_flux_fill_integration.py tests\test_runtime_surface_api.py tracked_tests\test_flux_fill_t5_gc_policy.py -q
```

Covers:

- runtime-native memory policy
- explicit posture defaults/guardrails without automatic assembly rearrangement
- Flux Fill route/session sanity (v3 greenfield and legacy compatibility)
- disk-paged T5 adaptive GC cadence with critical-headroom fallback
- runtime-surface preview and completed-image API ownership
- transition isolation behavior

### 4. Worker-Centric SDXL Lifecycle / Queue-Boundary / Interrupt Regressions

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_assembly_w10b.py tests\test_sdxl_assembly_w10d.py tests\test_sdxl_outer_wiring_w10c.py tracked_tests\test_sdxl_assembly_w03_regression.py tracked_tests\test_sdxl_assembly_w10b_lifecycle_coordinator.py -q
.\venv\Scripts\python.exe -m pytest tracked_tests\test_sdxl_assembly_w04_regression.py -k "assembly_progress_callback_preserves_interrupt_processing_exception or assembly_progress_callback_throttles_full_memory_telemetry" -q
.\venv\Scripts\python.exe -m pytest tests\test_runtime_surface_api.py -k "runtime_surface_skip_action_interrupts_active_task" -q
```

Covers:

- prompt-only invalidation narrowing into `prompt_conditioning`
- same-stack warm patched-CLIP reuse on prompt / `clip_skip` changes
- queue-frozen route truth, slot continuity, and fail-closed admission
- SDXL assembly callback interrupt preservation for running-task `Skip`

### 5. W11 Auxiliary Worker Lifecycle And Upscale Routing

```powershell
.\venv\Scripts\python.exe -m pytest tracked_tests\test_w11_gan_upscale_worker.py tracked_tests\test_w11_upscale_route_contract.py -q
.\venv\Scripts\python.exe -m pytest tracked_tests\test_w11_remove_workers.py tests\test_internal_assets.py -q
.\venv\Scripts\python.exe -m pytest tests\test_flux_fill_integration.py -k "remove_object_with_engine_dispatches_mat_and_flux or removal_stage_persists_background_and_object_outputs or flux_fill_removal_stage" -q
.\venv\Scripts\python.exe -m pytest tracked_tests\test_w11_color_enhanced_upscale.py tracked_tests\test_w11_color_enhanced_upscale_smoke.py -q
.\venv\Scripts\python.exe -m pytest tracked_tests\test_sdxl_progress_callback_compatibility.py -q
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_assembly_w06.py -k "vae_encode_worker_transient_lifecycle or vae_encode_cache_hit_preserves_blend_mask" -q
```

Covers:

- worker-owned GAN load, infer, device detach, and teardown
- generic auxiliary admission/failure/release telemetry
- failure-path worker teardown and lease release
- scalar-only UI scale metadata caching with no retained model object
- absence of broad runtime/cache cleanup during GAN execution
- direct light upscale and GAN-first `super-upscale` tiled-refinement handoff
- backend-owned BGR/MAT load, infer, detach, teardown, and failure cleanup
- sequential BGR-then-MAT auxiliary leases with no model overlap
- direct BGR/MAT route dispatch and neutral RGBA/mask/image output contracts
- legacy transparent-RGBA source compositing onto white at the route file boundary
- truthful auxiliary progress text on direct BGR/MAT routes with no false Flux label
- MAT small-image and tiled-image behavior, including deterministic seed input
- model-registry asset resolution without legacy module-global model caches
- Flux removal remaining on the Flux Fill v3 adapter boundary
- Color Enhancement target validation, strict original-source SDXL policy,
  optional tab-local prompt semantics,
  warm-UNet/Lora reuse, request-local overlay cleanup, deterministic bucket
  selection, phase-stable undecimated color transplant, low-VRAM
  transformer-tile caps, CPU-side tile accumulation, and final output
  shape/range contracts

## Manual Acceptance Replay

These are recommended scenario checks for Flux Fill route ownership changes and
are especially useful before pushing to Colab:

### 1. Flux Warm-Reuse Replay

Run this exact UI sequence:

1. `Inpaint (cold)`
2. `Inpaint (prompt change)`
3. `Remove`

Expected result:

- prompt-changed inpaint still reuses the warm streaming Flux UNet spine
- Flux Fill remove completes without the previous prompt-conditioning crash
- pure Flux Fill remove stays on the reusable Flux-owned path rather than taking
  the generic MAT/BGR-style aggressive reclaim branch
- when switching from `Inpaint` to `Remove` on a slower frontend such as Colab,
  Generate resolves the visibly active image-input tab at submit time so the
  request does not fall back to `txt2img` because of a stale hidden tab state
- once submitted, the queued task keeps its own frozen requested route and
  removal goals, so later UI changes or later queued txt2img jobs do not
  retroactively change how that earlier task is executed

### 2. Combined Removal Replay

Run this exact UI sequence:

1. `Remove (bg remove + obj remove)`

Expected result for the W11b direct MAT path:

- the BGR worker completes teardown and releases its auxiliary lease before MAT
  admission begins
- the MAT worker completes and releases its own lease after receiving neutral
  image/mask arrays
- no `cleanup_memory(... unload_models=True/force_cache=True)` or main-family
  teardown runs for the BGR/MAT pair

Flux Fill object removal remains a separate main-family adapter path. Its
transition/preflight reconciliation is carried into W11e and is not inferred
from the BGR/MAT evidence above.

### 3. Colab Free Disk-Paged T5 Replay

Run this exact UI sequence:

1. `Inpaint (cold)`
2. `Inpaint (prompt change)`
3. `Remove`
4. `Remove (warm)`
5. `Remove (bg remove + obj remove)`

Expected result:

- cache-miss prompt conditioning still completes with `disk_paged_t5`
- the disk-paged T5 runtime defaults to periodic host GC and only tightens
  toward every-block collection if live RAM headroom drops into the critical
  band
- queue-frozen route identity and removal goals remain stable across the full
  mixed sequence, even if later queued work is txt2img

### 4. High-RAM Resident T5 Replay

Run this exact UI sequence after explicitly setting Flux Fill T5 posture to
`cpu_resident` on a machine that exposes the option:

1. `Inpaint (cold)`
2. `Inpaint (prompt change)`
3. `Inpaint (prompt change)`
4. Switch T5 posture back to `disk_paged`
5. `Inpaint (prompt change)`

Expected result:

- the first resident run logs `Loading eager T5 safetensors`
- later prompt-changed resident runs log `Reusing cached CPU-resident text encoder`
- process-aware telemetry shows a persistent resident footprint across the warm
  resident runs through `proc_rss`, and when the platform provides them,
  `proc_shared`, `proc_uss`, and `proc_pss`
- do not use the Colab System RAM chart or `ram_available` alone as proof that
  resident T5 is absent; those surfaces reflect available-memory accounting,
  while the authoritative resident-worker truth is the process-aware telemetry
- after switching back to `disk_paged`, the next request tears down the warm
  CPU-resident text encoder before disk-paged execution begins

### 5. SDXL Prompt-Only Warm Reuse And Skip Replay

Run this exact UI sequence on the streaming SDXL lane:

1. `Txt2Img or Inpaint (cold, with a CLIP-side LoRA stack if available)`
2. `Prompt change` while keeping checkpoint, LoRA stack, and route assets the same
3. `Prompt change` again
4. While the later run is sampling, press `Skip`

Expected result:

- prompt-only edits do not reload a clean UNet or trigger UNet-side LoRA prepatching
- same-stack prompt edits reuse the current warm patched CLIP slot instead of rebuilding CLIP from scratch
- `Skip` cleanly interrupts the current image without surfacing the prior callback error
- if later queued work exists, execution advances to the next queued item instead of continuing the skipped image

### 6. Color Enhancement Local Replay

Run on local assets with an original image and its previously generated GAN
upscale. Keep the selected base model and LoRA stack fixed across a normal SDXL
run and color-enhanced-upscale run.

Place the previously generated GAN result in `Color Enhancement Target`. It
must be at least as large as the source in both dimensions and is used only as
the wavelet high-frequency content donor.

Expected result:

- The route fails clearly when the color enhancement target is absent or smaller than
  the original image.
- No GAN admission/load/infer telemetry appears; `color_enhancement_target`
  reports the target dimensions.
- The color pass always reports `sampler=dpmpp_2m` and keeps `cfg=1.5`, but it
  inherits the user-selected scheduler and steps instead of hardcoding
  `beta/18`.
- Sampling progress produces no `progressbar() takes 3 positional arguments
  but 5 were given` errors.
- Empty `Upscale Prompt` produces empty positive conditioning; a supplied
  tab-local prompt is used instead of the main prompt.
- The main negative prompt is preserved.
- The selected warm UNet and pre-patched LoRA stack are reused unless the
  checkpoint or LoRA stack changes.
- The SDXL color pass always VAE-encodes the original image resized to the
  selected SDXL bucket. The color enhancement target is never a VAE source.
- The final image is contiguous HWC RGB `uint8` at GAN output dimensions.
- The result gallery receives only the newly generated
  `Color Enhancement`; the provided target is not saved again.
- `vae_encode_begin` reports the color route and exact bucket-shaped BB tensor;
  `vae_encode_attached` reports the live device/dtype; and
  `vae_encode_compute_complete` separates encode compute time from attach/eject
  with CUDA allocated/reserved/peak values.
- An already attached transient VAE uses the preloaded encode seam and does not
  re-enter `prepare_models_for_stage stage=vae_encode` during the encode call.
- The route enters the `diffusion` residency phase, not `upscale`; no
  `upscaler_model` is pinned for Color Enhancement.
- `spine_stream_latent_finite` appears before VAE decode. Any non-finite sampled
  latent or decoded pixel tensor fails before conversion/saving, with no
  `invalid value encountered in cast` warning and no edge-residual output.
- The color-enhanced image has no regular 32-pixel block lattice; color transfer
  remains smooth under a one-pixel source translation.

### 7. Tracked Route / Stage Smoke

```powershell
.\venv\Scripts\python.exe -m pytest tracked_tests\test_pipeline_routes.py tracked_tests\test_pipeline_stage_runtime.py tracked_tests\test_memory_residency.py -q
```

Covers:

- route-family selection
- stage runner execution contract
- memory residency dispatch smoke

### 8. Full Suite

```powershell
.\venv\Scripts\python.exe -m pytest tests\ --ignore=tests\test_bgr.py --ignore=tests\test_objr.py -q
```

Notes:

- `tests/test_bgr.py` and `tests/test_objr.py` remain outside the closure bundle
  because of the pre-existing `args_manager` argparse incompatibility.
- Treat this command as the broad regression sweep after the targeted matrix is
  already green.
- Remaining W11 scaffold modules exist as intentionally skipped placeholders
  until their slices land:
  `tracked_tests/test_w11_auxiliary_queue_preview.py`.

## Optional Benchmarks

These are evidence tools, not closure gates:

```powershell
.\venv\Scripts\python.exe tools\bench_sdxl_pinned_residency_matrix.py
.\venv\Scripts\python.exe tools\bench_sdxl_resident_lora_lifecycle.py --placement both
.\venv\Scripts\python.exe tools\bench_headless_gguf_txt2img.py
.\venv\Scripts\python.exe tools\bench_flux_fill_fp8_streaming.py
```
