# Model Catalogs

This folder holds normalized model-catalog JSON files for M06.

The app-facing goal is:
- keep source catalogs separate from runtime code
- support multiple catalog files at once
- normalize different provider formats into one runtime view
- keep thumbnails repo-owned and stable across Colab sessions

## Preset Catalogs and References

- `civitai_main_catalog.json` - committed preset CivitAI catalog
- `huggingface_main_catalog.json` - committed preset HuggingFace catalog
- `github_main_catalog.json` - committed preset GitHub Release catalog

These main preset JSON files are also the reference examples for committed catalog structure in this folder.

Committed preset/source catalogs live in this root folder. Runtime-generated writable catalogs live in `configs/model_catalogs/user/`, alongside installed-link state and personal catalogs.

## Source Catalogs vs Runtime Catalogs

Every catalog should declare its upstream provider when it is created.

For M06, the provider layer is currently:
- CivitAI catalogs
- HuggingFace catalogs
- GitHub Release catalogs
- GitHub Release catalogs
- local catalogs for installed-only assets

Private or personal catalogs may still fall under one of those providers. In other words:
- `private` / `personal` describe ownership and maintenance
- `source_provider` describes the actual download source and token behavior

Examples:
- a personal CivitAI catalog should still use `source_provider: "civitai"`
- a private HuggingFace catalog should still use `source_provider: "huggingface"`
- a GitHub Release catalog should use `source_provider: "github"`
- a discovered local-only model can use `source_provider: "local"`

The app should normalize them into one unified runtime index, but users can still maintain them as separate JSON files.

## Recommended Asset Layout

For clarity, model assets should be organized by model root first, then architecture, then subtype where that subtype is meaningful.

Checkpoint layout:
- `checkpoints/sd15/base/`
- `checkpoints/sdxl/base/`
- `checkpoints/sdxl/pony/`
- `checkpoints/sdxl/illustrious/`
- `checkpoints/sdxl/noob/`

GGUF model execution is not part of the supported product surface. Active
catalog entries use supported checkpoint and LoRA formats; generic
quantization naming may still be retained in historical or user-authored
metadata without creating a runtime GGUF route.

Recommended UNet layout:
- `unet/sdxl/base/`
- `unet/sdxl/pony/`
- `unet/sdxl/illustrious/`
- `unet/sdxl/noob/`

Recommended CLIP layout:
- `clip/sdxl/base/`
- `clip/sdxl/pony/`
- `clip/sdxl/illustrious/`
- `clip/sdxl/noob/`

LoRAs follow the same convention as checkpoints and UNet, except there is no Noob LoRA bucket:
- `loras/sd15/base/`
- `loras/sdxl/base/`
- `loras/sdxl/pony/`
- `loras/sdxl/illustrious/`

Embeddings are architecture-scoped but do not use Pony / Illustrious / Noob subtype buckets:
- `embeddings/sd15/`
- `embeddings/sdxl/`

VAE is also architecture-scoped only:
- `vae/sd15/`
- `vae/sdxl/`

For quantized SDXL workflows, the same SDXL VAE family may be shared across multiple models, including fp16 and other variants.

The `loras/sdxl/noob/` example from earlier drafts should be considered obsolete.

## Recommended Multi-Catalog Layout

Preset catalogs that should ship with the repo live in this root folder.

Runtime-generated user/private catalogs should live in the writable `configs/model_catalogs/user/` folder and be loaded alongside the preset catalogs.

Recommended runtime naming:
- active catalogs: `*.catalog.json`
- compatible user/runtime catalogs: `*_catalog.json`
- optional examples: `*.example.json`

The runtime loader ingests `*.catalog.json` and `*_catalog.json` files so user-maintained catalogs can keep a more descriptive filename style. Example files remain excluded.


Conceptually, the runtime view is built from:
1. one or more catalog JSON files
2. filesystem scanning of installed models
3. active download-job state

## Important Identity Rules

Do not use filenames alone as the canonical identity.

Each normalized entry should distinguish between:
- `id`: unique entry identity
- `architecture`: broad runtime family such as `sdxl` or `sd15`
- `sub_architecture`: organization subtype such as `base`, `pony`, `illustrious`, `noob`, or `none` when no subtype split is needed
- `asset_group_key`: shared family identity across variants
- source identity such as `source_provider` and `source_version_id`

This allows:
- installed models to be removed from the Available view
- quantized variants such as `Q8`, `Q5_K_M`, and `Q4_K_M` to share one thumbnail simply by pointing to the same `thumbnail_library_relative`
- Colab session downloads to disappear without breaking thumbnail lookup

## Thumbnail Strategy

Thumbnails are repo-owned assets, not runtime-fetched dependencies.

Recommended behavior:
- store `thumbnail_library_relative` as the authoritative thumbnail path
- when registering a new model, auto-generate that mirrored path from the model taxonomy and chosen slug
- if no specific thumbnail exists, store or resolve `thumbnails/default_0001.png`

The repo thumbnail tree should mirror the effective model tree wherever grouping is meaningful.

Examples:
- `thumbnails/checkpoints/sd15/`
- `thumbnails/checkpoints/sdxl/base/`
- `thumbnails/checkpoints/sdxl/pony/`
- `thumbnails/unet/sdxl/noob/`
- `thumbnails/loras/sdxl/illustrious/`
- `thumbnails/embeddings/sdxl/`
- `thumbnails/vae/sdxl/`

### Naming Convention

Human-friendly thumbnail filenames should use:
- `{code}_{slug}.png`

Where `code` is generated from taxonomy only when that taxonomy is actually meaningful for the model type:
- SD15 all types: `{architecture}_{model_type}`
- SDXL checkpoints / UNet / CLIP: `{architecture}_{sub_architecture}_{model_type}`
- SDXL LoRAs: `{architecture}_{sub_architecture}_{model_type}` with no `noob` LoRA bucket
- SDXL embeddings / VAE: `{architecture}_{model_type}`

Examples:
- `default_0001.png`
- `sd15_checkpoint_anything_v5.png`
- `sdxl_base_checkpoint_stoiqo.png`
- `sdxl_pony_lora_powerpuff.png`
- `sdxl_noob_unet_homoveritas.png`
- `sdxl_vae_sdxl_vae.png`

The actual persisted binding should come from `thumbnail_library_relative`, not just filename matching.

Normal users should not need to invent thumbnail labels manually. The app should auto-generate the code/filename stem from the model metadata and shared-family identity, while still allowing advanced/manual overrides later if needed.

Display names may be auto-generated from `name`, but catalogs can also override them when a cleaner UI label is helpful. A good default rule is: take `name`, remove the file extension, and replace underscores with spaces so distinct variants like `Q4_K_M`, `Q5_K_M`, and `Q8` remain visibly distinct in the UI.

If a user skips thumbnail selection entirely, the catalog entry should simply resolve to `thumbnails/default_0001.png`.

PNG is the preferred thumbnail format for M06 so the library can grow into metadata-bearing thumbnails later without another format migration.

## CivitAI Download Convention

For the Director's current CivitAI workflow, downloads are token-authenticated through `.env`.

Expected pattern:
- base URL: `https://civitai.com/api/download/models/`
- final URL: `https://civitai.com/api/download/models/{id}?token={CIVITAI_TOKEN}`

Normalized CivitAI entries should therefore typically declare:
- `source_provider: "civitai"`
- `token_required: true`
- a `source` entry with `token_env: "CIVITAI_TOKEN"`

## Legacy Filename Prefixes

Legacy prefixes such as:
- `SDXL_`
- `PONY_`
- `IL_`
- `Noob_`

should be treated as import hints, not part of the long-term canonical naming scheme.

Normalized entries should use the normalized `name` as the persisted filename field, and runtime taxonomy should come from metadata and folder structure rather than name prefixes.

## Registration States

`registration_state` tracks where an entry sits in the catalog lifecycle:
- `unregistered`: auto-discovered or draft metadata that still needs user review
- `locally_registered`: confirmed catalog entry without a remote download source
- `sourced_registered`: confirmed catalog entry backed by a remote source such as HuggingFace, GitHub, or CivitAI

## Key Fields in the Runtime Schema

Common runtime fields include:
- `id`
- `alias`
- `name`
- `display_name`
- `model_type`
- `architecture`
- `sub_architecture`
- `root_key`
- `relative_path` (optional for authoritative sourced catalogs; derived from `architecture`, `sub_architecture`, and `name` when omitted)
- `asset_group_key`
- `thumbnail_library_relative`
- `source_provider`
- `source_version_id`
- `registration_state`
- `visibility`
- `preset_managed`
- `token_required`
- `source`

## Planned Runtime Consumers

- `modules/model_download/catalog.py`
- `modules/model_download/policy.py`
- `modules/model_download/resolver.py`
- `modules/model_download/transport.py`
- `modules/model_download/orchestrator.py`
- future M06 runtime index / manager modules







