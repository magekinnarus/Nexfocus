# Runtime Validation

This file defines the canonical validation contract for the post-W11 runtime
model. Each section is labeled by the environment that can execute it.

Historical mission work reports still describe what was true when each work
order closed. Use the commands in this file for current validation and closure
evidence.

## Environment (local maintainer)

This check depends on the ignored maintainer `tests/` tree and project virtual
environment. It is not available in a normal fresh clone. Run it from a
maintainer worktree through the project virtual environment:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_validation_environment.py -q
```

Validated local W12 baseline:

- `transformers==4.44.2`
- `huggingface-hub==0.36.2`
- `tokenizers==0.19.1`
- `accelerate==1.13.0`

Hugging Face model downloads use a single-stream Python `GET` through the
project downloader. Requests append `download=true`, use the existing retry
logic, and write only a temporary `.downloading` file before finalizing. Aria2,
Hugging Face Hub, and `hf-xet` are not used for this path; interrupted HF
downloads are not resumed.

CivitAI and GitHub model downloads use Aria2 with 16 connections/splits.
Unknown generic URLs retain the conservative 4-connection Aria2 path.

Every cached or newly downloaded `.safetensors` file is checked at the shared
download boundary before it is accepted. The check reads only the bounded
safetensors JSON header; HTML/XML error responses and unresolved pointer files
are deleted instead of being passed to a model loader. Manifest-backed assets
may then continue to another declared source. The Flux FP16 T5 asset is
deliberately CivitAI-only: its exact expected size is 9,787,841,024 bytes, and
there is no automatic Hugging Face fallback. CivitAI API downloads append
`CIVITAI_TOKEN` when configured. A login/HTML redirect fails closed with a
clear token requirement instead of being passed to Aria2.

The `support_models` GitHub Release is now the primary source for the moved
startup/support assets. The asset manifests list the GitHub Release URL first
and retain the previous Hugging Face URL as a fallback where one exists. The
Colab initial SDXL VAE download also uses the release directly. The W12a checkpoint and three benchmark LoRAs
remain on Hugging Face because they are not present in the release yet.

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

## Search And Compile (local maintainer)

These commands depend on the ignored maintainer `tests/` tree and project
virtual environment. They are not fresh-clone/public gates.

Ownership/runtime audit search:

```powershell
rg -n "sdxl_runtime_owner|process_diffusion|runtime_family|execution_mode|flux_fill" backend modules tests
```

Compile sanity on the authoritative runtime surfaces:

```powershell
$fluxV3Files = Get-ChildItem backend\flux_fill_v3\*.py | ForEach-Object { $_.FullName }
.\venv\Scripts\python.exe -m py_compile @fluxV3Files backend\memory_governor.py backend\process_transition.py backend\resources.py backend\sdxl_runtime_policy.py backend\sdxl_streaming_runtime.py backend\sdxl_unified_runtime.py backend\staging_manager.py backend\sdxl_assembly\assembler.py backend\sdxl_assembly\director.py backend\sdxl_assembly\gateway.py backend\sdxl_assembly\cpu_text_encode_worker.py backend\sdxl_assembly\gpu_lora_worker.py backend\sdxl_assembly\gpu_text_encode_worker.py backend\sdxl_assembly\lifecycle_coordinator.py backend\sdxl_assembly\progress.py backend\sdxl_assembly\request_builder.py backend\sdxl_assembly\runtime_state.py modules\async_worker.py modules\objr_engine.py modules\parameter_registry.py modules\pipeline\inference.py modules\pipeline\routes.py modules\pipeline\tiled_refinement.py modules\runtime_surface_state.py modules\runtime_surface_api.py modules\task_state.py modules\ui_components\advanced_panel.py modules\ui_logic.py webui.py tests\test_validation_environment.py tests\test_memory_residency.py tests\test_pipeline_routes.py tests\test_pipeline_stage_runtime.py tests\test_sdxl_assembly_w12b_production_resident.py tests\test_sdxl_assembly_w12c_gpu_text.py tests\test_runtime_surface_api.py tests\test_sdxl_assembly_w10d.py tests\test_sdxl_outer_wiring_w10c.py
```

## Regression Matrix (local maintainer)

These commands use the ignored maintainer `tests/` tree and project virtual
environment. They are not clone-owned CI coverage.

### 1. Unified SDXL Runtime And Image-Input Handoff

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_unified_runtime.py tests\test_unified_runtime_handoff.py tests\test_async_worker_process_transition.py tests\test_default_pipeline_process_reset.py tests\test_unsupported_public_surfaces.py -q
```

Covers:

- standard unified SDXL route
- unified SDXL image-input route
- authoritative runtime handoff
- unsupported checkpoint boundary expectations
- process-transition and cleanup behavior
- tiled-refinement runtime dispatch and interrupt semantics

### 2. Unsupported Checkpoint Boundary

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_unsupported_public_surfaces.py -q
```

Covers:

- `.gguf` rejection at model selection, loader, and core boundaries
- no public GGUF execution owner

### 3. Runtime-Centered Memory / Hardware / Flux Fill / Runtime-Surface Sanity

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_memory_governor.py tests\test_w11_policy_simplification.py tests\test_async_worker_process_transition.py tests\test_flux_fill_v3.py tests\test_flux_fill_integration.py tests\test_runtime_surface_api.py tests\test_flux_fill_t5_gc_policy.py -q
```

Covers:

- runtime-native memory policy
- explicit posture defaults/guardrails without automatic assembly rearrangement
- Flux Fill route/session sanity (v3 native FP8/safetensors path)
- disk-paged T5 adaptive GC cadence with critical-headroom fallback
- runtime-surface preview and completed-image API ownership
- transition isolation behavior

### 4. Worker-Centric SDXL Lifecycle / Queue-Boundary / Interrupt Regressions

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_assembly_w10b.py tests\test_sdxl_assembly_w10d.py tests\test_sdxl_outer_wiring_w10c.py tests\test_sdxl_assembly_w12b_production_resident.py tests\test_sdxl_assembly_w12c_gpu_text.py -q
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_assembly_w04.py -k "assembly_progress_callback_preserves_interrupt_processing_exception or assembly_progress_callback_throttles_full_memory_telemetry" -q
.\venv\Scripts\python.exe -m pytest tests\test_runtime_surface_api.py -k "runtime_surface_skip_action_interrupts_active_task" -q
```

Covers:

- prompt-only invalidation narrowing into `prompt_conditioning`
- same-stack warm patched-CLIP reuse on prompt / `clip_skip` changes
- queue-frozen route truth, slot continuity, and fail-closed admission
- SDXL assembly callback interrupt preservation for running-task `Skip`

### 5. W11 Auxiliary Worker Lifecycle And Upscale Routing

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_w11_gan_upscale_worker.py tests\test_w11_upscale_route_contract.py -q
.\venv\Scripts\python.exe -m pytest tests\test_w11_remove_workers.py tests\test_internal_assets.py -q
.\venv\Scripts\python.exe -m pytest tests\test_flux_fill_integration.py -k "remove_object_with_engine_dispatches_mat_and_flux or removal_stage_persists_background_and_object_outputs or flux_fill_removal_stage" -q
.\venv\Scripts\python.exe -m pytest tests\test_w11_color_enhanced_upscale.py tests\test_w11_color_enhanced_upscale_smoke.py -q
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_progress_callback_compatibility.py -q
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_assembly_w06.py -k "vae_encode_worker_transient_lifecycle or vae_encode_cache_hit_preserves_blend_mask" -q
```

Covers:

- worker-owned GAN load, infer, device detach, and teardown
- generic auxiliary admission/failure/release telemetry
- failure-path worker teardown and lease release
- scalar-only UI scale metadata caching with no retained model object
- absence of broad runtime/cache cleanup during GAN execution
- direct light upscale and target-first `super-upscale` tiled-refinement handoff
- auxiliary-only route boundaries preserving an already-active SDXL/Flux major
  family without publishing a synthetic route-owned SDXL identity for plain
  `upscale`
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

## Manual Acceptance Replay (historical/manual)

These are recommended scenario checks for Flux Fill route ownership changes and
are especially useful before pushing to Colab. They require local hardware,
models, or UI interaction and are not fresh-clone executable gates:

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

### 6. SDXL W12b Auto/Streaming Resident Replay

Current field status (2026-07-15): the original five L4 `auto` resident runs,
resident outpaint, and the Issues10 `SDXL inpaint -> Flux inpaint -> SDXL
inpaint` round trip passed. The Director accepts W12b with the remaining field
items transferred rather than erased. The host-reclaim round trip moves to
W12c on Colab Free T4; final broad route and Skip replay moves to W12e after
the W12d workflow-plan correction. New runs
must continue to show truthful
`inpaint_assembly` / `outpaint_assembly` route IDs,
`spatial_compose_complete ... blend=morphological_sin2`, and run-local
`CUDA_Peak` values.

Retain this UI sequence as the W12e parity checklist; completed items need not
be repeated solely to reopen W12b:

1. `SDXL Assembly Posture = streaming`, `Txt2Img`
2. `SDXL Assembly Posture = auto`, `Txt2Img (cold)`
3. `Prompt change` while keeping checkpoint, LoRA stack, and route assets fixed
4. `One UNet+CLIP LoRA`, then repeat with the same stack
5. Remove the LoRA stack
6. One image-input route that requires transient VAE encode/decode
7. One accepted ControlNet route where assets permit
8. `Color Enhancement`
9. `Super-Upscale`
10. Press `Skip` during a resident run
11. Switch to Flux or trigger an explicit full release

Expected result:

- `auto` resolves to resident UNet on L4, while `streaming` forces streaming.
- Each run emits matched `[SDXL RUN BEGIN]` and `[SDXL RUN END]` records with
  the same correlation ID.
- Resident cold load, prompt-only warm reuse, LoRA same-stack reuse, LoRA
  removal, transient VAE attach/detach, ControlNet coexistence, skip/interrupt,
  and final release are visible in console telemetry.
- On the SDXL-to-Flux transition, the departing resident spine releases before
  Flux activation. If a provider returns HTML/XML under a `.safetensors`
  filename, the downloader rejects and deletes it instead of failing later
  inside prompt encoding. For the CivitAI-only FP16 T5 asset, an authentication
  redirect must report the `CIVITAI_TOKEN` requirement; it must not fall back to
  Hugging Face automatically.
- On supported Colab Linux profiles, each checkpoint/family switch reports
  `checkpoint_switch ... trim_host=True` followed by
  `cleanup ... trimmed=True ... proc_rss_before=... proc_rss_after=...`.
  Repeat `SDXL -> Flux -> SDXL`; the next-family entry RSS must fall materially
  after each departing-family release and must not show a stepwise baseline
  increase across the round trip. Same-stack SDXL warm reuse remains retained.
- Report CPU RSS, CUDA allocated/reserved/peak, output success/path, resident
  spine retention/release, and any failure/interrupt status in the same
  issue/outcome style as `.agent/temp/P4-M18-W11e_issues4.md`.
- Full Colab Free T4 stress, including the explicit GPU text composition and
  three-LoRA/three-ControlNet headroom where assets permit, begins in W12c.

### 6.1 SDXL W12c GPU-Text Colab Free T4 Replay

Use a Colab Free T4-class session with approximately 12.7 GB host RAM and
explicitly select `gpu_text` (`resident_unet_gpu_text`). Do not count an L4 or a
CPU-text run as field acceptance for this composition.

Required sequence:

1. Cold Txt2Img without LoRA.
2. Prompt-only change with the same checkpoint and stack.
3. One preflighted LoRA with a proven non-empty CLIP patch dictionary; repeat it.
4. CLIP-only change, UNet-only change, combined change, and LoRA removal.
5. A requested CLIP LoRA that resolves to zero actual patches and bypasses compilation.
6. Inpaint with transient VAE encode/decode.
7. Three LoRAs plus three ControlNets, including structural and PuLID where assets permit.
8. `SDXL -> Flux disk-paged T5 -> SDXL` using the Issues10 transition sequence.
9. Explicit full release.

Expected evidence:

- one authoritative CUDA CLIP-L/CLIP-G owner, zero retained CPU/GPU clean
  shadow, and separate resident UNet/text byte inventories;
- compile baseline, peak, peak delta, final allocation, patch count, cleared
  patch count, and zero retained host-pinned adapter bytes;
- prompt/same-stack reuse, side-specific LoRA invalidation, checkpoint-backed
  in-place CLIP restoration, and zero-patch bypass truth;
- resident UNet + GPU text coexistence through transient VAE and selected CN
  windows without hidden eviction, fallback, or OOM;
- if a same-family request is assembly-ineligible and enters the legacy SDXL
  runtime, `assembly_route_legacy_transition` must precede legacy checkpoint
  loading and the retained assembly UNet/GPU-text inventories must be released;
- an Outpaint request with ControlNet mixing disabled must ignore populated
  hidden CN slot values and must not report CPDS or another inactive CN as its
  reason for leaving the assembly lane;
- materially lower process RSS than the comparable W12b CPU-text composition;
- `checkpoint_switch ... trim_host=True` and cleanup
  `proc_rss_before`/`proc_rss_after` evidence with no stepwise family-transition
  RSS growth; and
- no Hugging Face fallback for the CivitAI-only FP16 Flux T5.

### 6.2 SDXL W12d Queue-Frozen Workflow Plan and ControlNet Overlay Replay

W12d is a pipeline-wide outer-layer correction, not a GPU-text worker feature.
Run the local truth-table suite across streaming, resident CPU text, and
resident GPU text. Then run the narrow physical replay on Colab Free T4 with
`gpu_text` selected.

Required truth table:

1. Normal Generate surface with hidden populated CN controls -> `txt2img`, CN overlay off.
2. ControlNet tab with one supported active slot -> `txt2img`, CN overlay on.
3. ControlNet tab with inpaint/outpaint mixing checkboxes set -> still `txt2img`, CN overlay determined only by active slots.
4. Inpaint tab, mixing off, hidden populated slots -> `inpaint`, CN overlay off.
5. Inpaint tab, mixing on, one supported slot -> `inpaint`, CN overlay on.
6. Outpaint tab, mixing off, hidden populated slots -> `outpaint`, CN overlay off.
7. Outpaint tab, mixing on, one supported slot -> `outpaint`, CN overlay on.
8. Remove, Upscale, Color Enhancement, and Super-Upscale surfaces -> selected base route, CN overlay off regardless of hidden slot values.

Expected evidence:

- one immutable plan record identifies base route, route family, overlay
  activation/source, literal active slots/types, and ordered stages;
- later route, asset, admission, and transition records agree with that plan;
- overlay-off inpaint/outpaint plans contain no ControlNet support,
  structural-preprocess, or contextual-preprocess stage;
- inactive slots cause no support-asset resolution/download, assembly
  rejection, transition, or active-CN telemetry;
- queued execution remains unchanged after later UI tab, checkbox, image, or
  slot edits;
- an actually unsupported active CN request still fails closed and preserves
  release-before-legacy-load behavior; and
- the T4 replay shows no hidden-CPDS bypass, duplicate UNet/CLIP owner, or
  route-induced OOM.

W12d evidence may satisfy the matching route/transition items in W12c, but
does not replace W12c's positive CLIP-LoRA, RSS comparison,
`SDXL -> Flux -> SDXL`, or explicit full-release evidence. Broad Color
Enhancement, Super-Upscale, Skip, and cross-route parity remain W12e.

### 7. Color Enhancement Local Replay

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

### 8. Migrated Route / Stage Smoke

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_pipeline_routes.py tests\test_pipeline_stage_runtime.py tests\test_memory_residency.py -q
```

Covers:

- route-family selection
- stage runner execution contract
- memory residency dispatch smoke

### 9. Full Suite

```powershell
.\venv\Scripts\python.exe -m pytest tests\ --ignore=tests\test_bgr.py --ignore=tests\test_objr.py -q
```

Notes:

- `tests/test_bgr.py` and `tests/test_objr.py` remain outside the closure bundle
  because of the pre-existing `args_manager` argparse incompatibility.
- Treat this command as the broad regression sweep after the targeted matrix is
  already green.
- At the W13f candidate, this command reports `842 passed, 12 failed, 3
  warnings`. The exact 12 node IDs reproduce against the untouched pre-W13f
  `cd27bef` source and are ignored maintainer-fixture debt: one contextual
  tuple-shape fixture, two pinned-loader allocation fixtures, four superseded
  W07-W09 assembly eligibility/slot fixtures, and five workflow-plan-free
  super-upscale fixtures. They are not fresh-clone/public gates. See the local
  ignored W13f report for exact node IDs and baseline replay evidence.

## W13d Legacy-Surface Quarantine Checks

W13d retired the public `tracked_tests/` and `tools/` collections along with the GGUF headless runner and its callers, following the historical W13c Flux surface quarantine. These checks verify the public-tree boundary without treating the ignored `.legacy_reference/` archive as runtime source:

The required isolated Flux prompt-conditioning cache-miss worker is now owned
by `backend/flux_fill_v3/prompt_conditioning_artifact_worker.py`; it is not a
replacement general-purpose tools collection.

```powershell
git ls-files backend/flux backend/flux_fill_v2 modules/gguf_headless_runner.py tracked_tests tools
rg -n -P "backend\.flux(?!_fill_v3)|backend/flux/(?!fill_v3)|backend\.flux_fill_v2|backend/flux_fill_v2|modules/gguf_headless_runner|tracked_tests|tools[\\/]" backend modules tests
git ls-files .legacy_reference
git check-ignore -v .legacy_reference/P4-M18-W13/backend/flux/__init__.py
```

## W13e Validation Surface Classification

The existing validation surface is intentionally split by repository policy:

| Surface | Classification | Boundary |
|---|---|---|
| Environment check and the `Search And Compile` command | **local maintainer** | They reference the ignored `tests/` owners and use the maintainer virtual environment. |
| Regression Matrix sections 1-9 | **local maintainer** | The canonical tests remain ignored local maintainer tests; they are not clone-owned CI coverage. |
| Manual Acceptance Replay sections 1-8 | **historical/manual** | These are hardware, model, and UI replay instructions rather than fresh-clone executable gates. |
| W13d Legacy-Surface Quarantine Checks above | **local maintainer** | The `rg` command includes the ignored local `tests/` tree; the Git inventory subcommands remain useful public checks. |
| Commands below | **fresh-clone/public** | Every repository path named by the command is tracked in the integrated candidate, or is an intentionally absent ignored path tested by Git metadata. |

The `.agent/` tree contains local maintainer governance and is intentionally
absent from a fresh clone. Its absence is expected and must not be repaired by
publishing project-governance documents.

### Fresh-clone/public W13e checks

Run these from the root of a clean clone of the integrated candidate. The
Python checks may use an external compatible Python environment (including a
maintainer virtual environment), but the clone must remain the working directory
and import root. `-B` prevents validation from creating bytecode caches:

```powershell
git status --short
git ls-files tracked_tests tools modules/gguf_headless_runner.py .legacy_reference
git grep -n -P "backend\.flux(?!_fill_v3)|backend/flux/(?!fill_v3)|backend\.flux_fill_v2|backend/flux_fill_v2|modules/gguf_headless_runner" -- backend modules webui.py
git grep -n -E "from tools|import tools|generate_flux_t5_fp16_stream_artifact" -- backend modules webui.py
git ls-files --error-unmatch backend/flux_fill_v3/prompt_conditioning_artifact_worker.py backend/flux_fill_v3/t5_worker.py validation.md
git check-ignore -v .legacy_reference/P4-M18-W13/backend/flux/__init__.py
python -B -c "import subprocess, tokenize; paths=[p for p in subprocess.check_output(['git','ls-files','-z','--','*.py']).decode().split(chr(0)) if p]; [compile(tokenize.open(path).read(), path, 'exec') for path in paths]; print('tracked_python_compiled',len(paths))"
python -B -c "from pathlib import Path; from backend.flux_fill_v3 import prompt_conditioning_artifact_worker as worker; assert worker.REPO_ROOT == Path.cwd(); print('worker_import_ok', worker.REPO_ROOT)"
```

Expected results are an empty status; exit code 1 with no output from each
retired-path/import `git grep`; tracked-path proof for the maintained worker and
validation document; an ignore-rule match for the quarantine path; and a
successful compile/import result. The clone must not contain `.agent/`,
`.legacy_reference/`, `tests/`, `tools/`, or generated bytecode caches.

## W13f Residual Public-Surface Retirement

W13f removes the nine tracked historical Flux files, the unsupported tracked
GGUF execution stack, and the three acceleration presets. Their byte-exact
copies are retained only in the ignored local quarantine
`.legacy_reference/P4-M18-W13/`. The ignored local glass harness under
`backend/gguf/` is not a tracked product owner and is deliberately outside the
retirement disposition.

From a clean clone of the integrated candidate, these public-tree proofs must
produce no output:

```powershell
git ls-files archived backend/gguf modules/pipeline/gguf_runner.py presets/lcm.json presets/lightning.json presets/lightning_8step.json
git ls-files .legacy_reference .agent
git grep -n -E "backend\.gguf|modules\.pipeline\.gguf_runner|sample_lcm|path_loras_(lcm|lightning)|gguf>=|presets/(lcm|lightning)" -- backend modules configs requirements_versions.txt presets
```

The only supported `.gguf` references are fail-closed boundary checks for
selection/loading, plus the Flux Fill v3 text-encoder rejection. Confirm that
boundary and the absence of active owners with:

```powershell
git grep -n -i "gguf" -- backend modules configs requirements_versions.txt
git grep -n -E "from backend\.gguf|import backend\.gguf|gguf_runner|sample_lcm|opModelSamplingDiscrete|path_loras_lcm|path_loras_lightning" -- backend modules configs requirements_versions.txt
```

The first command is reviewed for boundary-only matches; the second must be
empty. The generic upstream `ldm_patched.modules.model_sampling.ModelSamplingDiscrete`
schedule patch in `modules/patch_precision.py` is retained precision
infrastructure, not the retired `modules.core.opModelSamplingDiscrete` LCM
shortcut. Re-run direct, lazy, dynamic, route, setup/dependency, validation,
documentation, and test searches from the maintainer worktree when updating a
candidate. Tokenizer vocabulary and historical update-log prose are data, not
runtime feature owners.

The focused maintained regression set is:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_unsupported_public_surfaces.py tests\test_sdxl_unified_runtime.py tests\test_unified_runtime_handoff.py tests\test_async_worker_process_transition.py tests\test_default_pipeline_process_reset.py tests\test_memory_governor.py tests\test_pipeline_routes.py tests\test_pipeline_stage_runtime.py tests\test_sdxl_assembly_w12b_production_resident.py tests\test_sdxl_assembly_w12c_gpu_text.py tests\test_flux_fill_v3.py tests\test_flux_fill_integration.py tests\test_runtime_surface_api.py -q
```

The retained preset, model-selection, process-transition, ordinary
scheduler-policy, and unsupported-boundary set is:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_preset_selection.py tests\test_config_dropdown_resolution.py tests\test_ui_logic_model_filtering.py tests\test_model_api.py tests\test_model_manager.py tests\test_model_catalog_index.py tests\test_model_registry.py tests\test_model_resolution.py tests\test_model_thumbnails.py tests\test_process_transition.py tests\test_resident_lora_lifecycle.py tests\test_w11_policy_simplification.py tests\test_internal_assets.py tests\test_unsupported_public_surfaces.py -q
```

Compile every current tracked Python source with `python -B` and run the
retained local collection command from the W13e section. W13f is not finally
accepted until those checks are repeated from a clean isolated clone of the
resulting integrated commit; a dirty CM candidate is integration-ready only.
