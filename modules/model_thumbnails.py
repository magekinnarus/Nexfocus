from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

import modules.config as config
import modules.model_taxonomy as model_taxonomy
from modules.model_download.spec import ModelCatalogEntry
from modules.util import LANCZOS


THUMBNAIL_EXTENSION = '.png'

THUMBNAIL_ROOT_KEY_MAP = {
    'checkpoints': 'checkpoints',
    'loras': 'loras',
    'embeddings': 'embeddings',
    'vae': 'vae',
    'unet': 'unet',
    'clip': 'clip',
}


@dataclass(frozen=True)
class ThumbnailResolution:
    relative_path: str
    absolute_path: str
    exists: bool
    source: str


def _normalize_relative_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).replace('\\', '/').strip().strip('/')
    return normalized or None


def _strip_thumbnail_prefix(relative_path: str) -> str:
    normalized = _normalize_relative_path(relative_path) or ''
    if normalized.startswith('thumbnails/'):
        return normalized[len('thumbnails/'):]
    if normalized == 'thumbnails':
        return ''
    return normalized


def _normalize_root_key(root_key: str | None) -> str:
    if root_key is None:
        return 'models'
    normalized = str(root_key).strip().lower()
    return THUMBNAIL_ROOT_KEY_MAP.get(normalized, normalized or 'models')


def _default_model_type(root_key: str | None) -> str:
    return {
        'checkpoints': 'checkpoint',
        'loras': 'lora',
        'embeddings': 'embedding',
        'vae': 'vae',
        'unet': 'unet',
        'clip': 'clip',
    }.get(_normalize_root_key(root_key), 'model')


def _normalize_model_type(model_type: str | None, root_key: str | None = None) -> str:
    normalized = str(model_type or '').strip().lower()
    if normalized:
        return normalized
    return _default_model_type(root_key)


def _uses_sub_architecture(architecture: str | None, model_type: str) -> bool:
    if architecture != model_taxonomy.ARCHITECTURE_SDXL:
        return False
    return model_type in {'checkpoint', 'unet', 'clip', 'lora'}


def _effective_sub_architecture(
    architecture: str | None,
    sub_architecture: str | None,
    *,
    model_type: str,
) -> str | None:
    normalized_architecture = model_taxonomy.normalize_architecture(architecture)
    normalized_sub_architecture = model_taxonomy.normalize_sub_architecture(
        sub_architecture,
        architecture=normalized_architecture,
    )

    if not _uses_sub_architecture(normalized_architecture, model_type):
        return None
    if normalized_sub_architecture == model_taxonomy.SUB_ARCHITECTURE_NOOB and model_type == 'lora':
        return model_taxonomy.SUB_ARCHITECTURE_ILLUSTRIOUS
    return normalized_sub_architecture or model_taxonomy.SUB_ARCHITECTURE_BASE


def _slugify(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).replace('\\', '/').strip()
    if not text:
        return None
    stem = Path(text.split('/')[-1]).stem
    normalized = re.sub(r'[^a-z0-9]+', '_', stem.lower())
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    return normalized or None


def _default_slug(entry: ModelCatalogEntry, slug: str | None = None) -> str:
    candidates = [slug]
    if entry.asset_group_key:
        candidates.append(str(entry.asset_group_key).split('.')[-1])
    candidates.extend([entry.alias, entry.name])
    for candidate in candidates:
        normalized = _slugify(candidate)
        if normalized:
            return normalized
    return 'model'


def get_thumbnail_library_root() -> Path:
    return Path(config.get_model_thumbnail_directory()).resolve()


def get_default_thumbnail_relative_path() -> str:
    return config.get_default_thumbnail_relative_path()


def get_default_thumbnail_path() -> Path:
    return resolve_thumbnail_absolute_path(get_default_thumbnail_relative_path())


def get_thumbnail_size() -> int:
    return config.get_model_thumbnail_size()


def resolve_thumbnail_absolute_path(relative_path: str | os.PathLike[str]) -> Path:
    normalized = _normalize_relative_path(relative_path)
    if normalized is None:
        raise ValueError('Thumbnail relative path is required')
    return get_thumbnail_library_root() / _strip_thumbnail_prefix(normalized)


def build_thumbnail_code(
    *,
    architecture: str | None,
    sub_architecture: str | None,
    model_type: str | None,
    root_key: str | None = None,
) -> str:
    normalized_architecture = model_taxonomy.normalize_architecture(architecture) or 'unknown'
    normalized_model_type = _normalize_model_type(model_type, root_key=root_key)
    normalized_sub_architecture = _effective_sub_architecture(
        normalized_architecture,
        sub_architecture,
        model_type=normalized_model_type,
    )

    parts = [normalized_architecture]
    if normalized_sub_architecture:
        parts.append(normalized_sub_architecture)
    parts.append(normalized_model_type)
    return '_'.join(parts)


def build_thumbnail_filename(
    *,
    architecture: str | None,
    sub_architecture: str | None,
    model_type: str | None,
    slug: str,
    root_key: str | None = None,
) -> str:
    code = build_thumbnail_code(
        architecture=architecture,
        sub_architecture=sub_architecture,
        model_type=model_type,
        root_key=root_key,
    )
    return f'{code}_{_slugify(slug) or "model"}{THUMBNAIL_EXTENSION}'


def build_thumbnail_relative_path(
    *,
    root_key: str | None,
    architecture: str | None,
    sub_architecture: str | None,
    model_type: str | None,
    slug: str,
) -> str:
    normalized_architecture = model_taxonomy.normalize_architecture(architecture) or 'unknown'
    normalized_model_type = _normalize_model_type(model_type, root_key=root_key)
    normalized_sub_architecture = _effective_sub_architecture(
        normalized_architecture,
        sub_architecture,
        model_type=normalized_model_type,
    )

    parts = ['thumbnails', _normalize_root_key(root_key), normalized_architecture]
    if normalized_sub_architecture:
        parts.append(normalized_sub_architecture)
    parts.append(
        build_thumbnail_filename(
            architecture=normalized_architecture,
            sub_architecture=normalized_sub_architecture,
            model_type=normalized_model_type,
            slug=slug,
            root_key=root_key,
        )
    )
    return '/'.join(parts)


def build_thumbnail_relative_path_for_entry(entry: ModelCatalogEntry, slug: str | None = None) -> str:
    return build_thumbnail_relative_path(
        root_key=entry.root_key,
        architecture=entry.architecture,
        sub_architecture=entry.sub_architecture,
        model_type=entry.model_type,
        slug=_default_slug(entry, slug=slug),
    )


def resolve_thumbnail(entry: ModelCatalogEntry, slug: str | None = None) -> ThumbnailResolution:
    if entry.thumbnail_library_relative:
        configured_relative = _normalize_relative_path(entry.thumbnail_library_relative)
        if configured_relative is not None:
            configured_absolute = resolve_thumbnail_absolute_path(configured_relative)
            if configured_absolute.is_file():
                return ThumbnailResolution(
                    relative_path=configured_relative,
                    absolute_path=str(configured_absolute),
                    exists=True,
                    source='catalog',
                )

    generated_relative = build_thumbnail_relative_path_for_entry(entry, slug=slug)
    generated_absolute = resolve_thumbnail_absolute_path(generated_relative)
    if generated_absolute.is_file():
        return ThumbnailResolution(
            relative_path=generated_relative,
            absolute_path=str(generated_absolute),
            exists=True,
            source='generated',
        )

    default_relative = get_default_thumbnail_relative_path()
    default_absolute = get_default_thumbnail_path()
    return ThumbnailResolution(
        relative_path=default_relative,
        absolute_path=str(default_absolute),
        exists=default_absolute.is_file(),
        source='default',
    )


def _open_source_image(source: str | os.PathLike[str] | Image.Image | Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    if source is None:
        raise ValueError('Thumbnail source image is required')
    with Image.open(source) as image:
        return image.copy()


def _center_crop(image: Image.Image, ratio: float = 1.6) -> Image.Image:
    width, height = image.width, image.height
    current_ratio = width / height

    if current_ratio > ratio:
        # Image is wider than target ratio
        new_width = int(ratio * height)
        offset = (width - new_width) // 2
        return image.crop((offset, 0, offset + new_width, height))
    else:
        # Image is taller than target ratio
        new_height = int(width / ratio)
        offset = (height - new_height) // 2
        return image.crop((0, offset, width, offset + new_height))


def persist_thumbnail_image(
    source: str | os.PathLike[str] | Image.Image | Any,
    *,
    target_relative_path: str | None = None,
    entry: ModelCatalogEntry | None = None,
    slug: str | None = None,
    size: int | None = None,
) -> ThumbnailResolution:
    if target_relative_path is None:
        if entry is None:
            raise ValueError('Either target_relative_path or entry is required')
        target_relative_path = build_thumbnail_relative_path_for_entry(entry, slug=slug)

    normalized_relative = _normalize_relative_path(target_relative_path)
    if normalized_relative is None:
        raise ValueError('Thumbnail target path is required')

    output_width = int(size or get_thumbnail_size())
    output_height = int(output_width / 1.6)
    output_path = resolve_thumbnail_absolute_path(normalized_relative)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_image = _open_source_image(source)
    try:
        processed = _center_crop(source_image.convert('RGBA'), ratio=1.6).resize(
            (output_width, output_height),
            resample=LANCZOS,
        )
        processed.save(output_path, format='PNG')
    finally:
        source_image.close()

    return ThumbnailResolution(
        relative_path=normalized_relative,
        absolute_path=str(output_path),
        exists=output_path.is_file(),
        source='persisted',
    )
