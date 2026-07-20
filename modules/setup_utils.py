import os
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from modules import config, model_registry
from modules.model_download.runtime import download_file

vae_approx_filenames = [
    ('taesdxl_decoder.pth',
     'https://github.com/magekinnarus/Fooocus_Nex/releases/download/support_models/taesdxl_decoder.pth'),
    ('taef1_decoder.pth',
     'https://github.com/magekinnarus/Fooocus_Nex/releases/download/support_models/taef1_decoder.pth'),
]


def _ensure_assets(label, assets, progress=False):
    assets = list(assets or [])
    if not assets:
        return

    start = time.perf_counter()
    print(f'[Startup] Ensuring {label} assets ({len(assets)}) ...')
    for asset in assets:
        model_registry.ensure_asset(asset['id'], progress=progress)
    print(f'[Startup] Ensured {label} assets in {time.perf_counter() - start:.2f}s')


def _ensure_internal_assets(category, progress=False):
    _ensure_assets(
        f'internal {category}',
        sorted(model_registry.list_assets(category=category, internal_only=True), key=lambda item: item['id']),
        progress=progress,
    )


def _ensure_guidance_assets(progress=False):
    for channel in ('Structural', 'Contextual'):
        _ensure_assets(
            f'{channel.lower()} guidance',
            sorted(model_registry.list_assets(channel=channel), key=lambda item: item['id']),
            progress=progress,
        )


def _ensure_startup_support_assets(progress=False):
    _ensure_internal_assets('upscale', progress=progress)
    _ensure_internal_assets('removal', progress=progress)
    _ensure_guidance_assets(progress=progress)
    model_registry.ensure_asset('inpaint.flux_fill.empty_conditioning', progress=progress)


def _resolve_startup_download_url(url: str) -> str:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    if not host.endswith('civitai.com'):
        return url

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get('token'):
        return url

    token = os.getenv('CIVITAI_TOKEN', '').strip()
    if not token:
        return url

    query['token'] = token
    return urlunparse(parsed._replace(query=urlencode(query)))


def _download_checkpoint_targets(checkpoint_downloads):
    downloaded_any = False
    for file_name, url in checkpoint_downloads.items():
        target_path = os.path.join(
            config.get_preferred_asset_root_path('checkpoints', file_name=file_name),
            file_name,
        )
        missing_before_download = not os.path.exists(target_path)
        try:
            download_file(
                url=_resolve_startup_download_url(url),
                model_dir=config.get_preferred_asset_root_path('checkpoints', file_name=file_name),
                file_name=file_name,
            )
            if missing_before_download and os.path.exists(target_path):
                downloaded_any = True
        except Exception as e:
            print(f'[Startup] Error downloading checkpoint {file_name}: {e}')
    return downloaded_any


def download_models(
    default_model,
    checkpoint_downloads,
    embeddings_downloads,
    lora_downloads,
    vae_downloads,
    upscale_downloads,
    *,
    include_vae_approx=True,
    validate_checkpoint_dirs=True,
    include_startup_support_assets=True,
    log_prefix='[Startup]',
):
    from modules.util import get_file_from_folder_list

    overall_start = time.perf_counter()
    downloaded_user_visible_assets = False

    if include_vae_approx:
        for file_name, url in vae_approx_filenames:
            try:
                download_file(url=url, model_dir=config.path_vae_approx, file_name=file_name)
            except Exception as e:
                print(f'{log_prefix} Error downloading vae_approx {file_name}: {e}')

    if checkpoint_downloads:
        checkpoint_start = time.perf_counter()
        downloaded_user_visible_assets = _download_checkpoint_targets(checkpoint_downloads) or downloaded_user_visible_assets
        print(f'{log_prefix} Checkpoint downloads completed in {time.perf_counter() - checkpoint_start:.2f}s')

    if validate_checkpoint_dirs:
        # Check if any model exists in checkpoints
        model_found = False
        for folder in config.paths_checkpoints:
            if os.path.isdir(folder):
                if any(f.endswith(('.safetensors', '.ckpt')) for f in os.listdir(folder)):
                    model_found = True
                    break

        if not model_found:
            print('No checkpoint models found in your checkpoints directories.')
            print('Please add at least one model to your checkpoints folder to start generating.')

    # Embeddings, Loras, VAE downloads (optional, kept if explicitly in config)
    for file_name, url in embeddings_downloads.items():
        target_path = os.path.join(config.path_embeddings, file_name)
        missing_before_download = not os.path.exists(target_path)
        try:
            download_file(url=url, model_dir=config.path_embeddings, file_name=file_name)
            if missing_before_download and os.path.exists(target_path):
                downloaded_user_visible_assets = True
        except Exception as e:
            print(f'{log_prefix} Error downloading embedding {file_name}: {e}')
    for file_name, url in lora_downloads.items():
        preferred_root = config.paths_loras[0]
        existing_path = get_file_from_folder_list(file_name, [preferred_root])
        if os.path.exists(existing_path):
            continue
        try:
            download_file(url=url, model_dir=preferred_root, file_name=file_name)
            if os.path.exists(os.path.join(preferred_root, file_name)):
                downloaded_user_visible_assets = True
        except Exception as e:
            print(f'{log_prefix} Error downloading LoRA {file_name}: {e}')
    for file_name, url in vae_downloads.items():
        preferred_root = config.get_preferred_asset_root_path('vae', file_name=file_name)
        target_path = os.path.join(preferred_root, file_name)
        missing_before_download = not os.path.exists(target_path)
        try:
            download_file(url=url, model_dir=preferred_root, file_name=file_name)
            if missing_before_download and os.path.exists(target_path):
                downloaded_user_visible_assets = True
        except Exception as e:
            print(f'{log_prefix} Error downloading VAE {file_name}: {e}')

    if include_startup_support_assets:
        # Front-load all support-model assets so the UI does not need to trigger them later.
        _ensure_startup_support_assets(progress=False)

    # Keep preset/config-defined entries as additive custom downloads rather than the source of truth.
    for file_name, url in upscale_downloads.items():
        preferred_root = config.get_preferred_asset_root_path('upscale_models', file_name=file_name)
        target_path = os.path.join(preferred_root, file_name)
        missing_before_download = not os.path.exists(target_path)
        try:
            download_file(url=url, model_dir=preferred_root, file_name=file_name)
            if missing_before_download and os.path.exists(target_path):
                downloaded_user_visible_assets = True
        except Exception as e:
            print(f'{log_prefix} Error downloading upscale model {file_name}: {e}')

    print(f'{log_prefix} download_models work completed in {time.perf_counter() - overall_start:.2f}s')
    return default_model, checkpoint_downloads, downloaded_user_visible_assets


def download_preset_models(default_model, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads, upscale_downloads):
    return download_models(
        default_model,
        checkpoint_downloads,
        embeddings_downloads,
        lora_downloads,
        vae_downloads,
        upscale_downloads,
        include_vae_approx=False,
        validate_checkpoint_dirs=False,
        include_startup_support_assets=False,
        log_prefix='[Preset]',
    )
