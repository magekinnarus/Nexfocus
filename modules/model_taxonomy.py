import os
from dataclasses import dataclass

ARCHITECTURE_SD15 = 'sd15'
ARCHITECTURE_SDXL = 'sdxl'
DEFAULT_ARCHITECTURE = ARCHITECTURE_SDXL
DEFAULT_COMPATIBILITY_FAMILY = DEFAULT_ARCHITECTURE

SUB_ARCHITECTURE_NONE = 'none'
SUB_ARCHITECTURE_BASE = 'base'
SUB_ARCHITECTURE_PONY = 'pony'
SUB_ARCHITECTURE_ILLUSTRIOUS = 'illustrious'
SUB_ARCHITECTURE_NOOB = 'noob'

SDXL_SUB_ARCHITECTURES = (
    SUB_ARCHITECTURE_BASE,
    SUB_ARCHITECTURE_PONY,
    SUB_ARCHITECTURE_ILLUSTRIOUS,
    SUB_ARCHITECTURE_NOOB,
)

ARCHITECTURE_ALIASES = {
    'sd 1.5': ARCHITECTURE_SD15,
    'sd1.5': ARCHITECTURE_SD15,
    'sd15': ARCHITECTURE_SD15,
    'sdxl': ARCHITECTURE_SDXL,
    'xl': ARCHITECTURE_SDXL,
}

SUB_ARCHITECTURE_ALIASES = {
    '': None,
    'base': SUB_ARCHITECTURE_BASE,
    'default': SUB_ARCHITECTURE_BASE,
    'general': SUB_ARCHITECTURE_BASE,
    'il': SUB_ARCHITECTURE_ILLUSTRIOUS,
    'illustrious': SUB_ARCHITECTURE_ILLUSTRIOUS,
    'noob': SUB_ARCHITECTURE_NOOB,
    'none': SUB_ARCHITECTURE_NONE,
    'null': SUB_ARCHITECTURE_NONE,
    'pony': SUB_ARCHITECTURE_PONY,
    'sdxl': SUB_ARCHITECTURE_BASE,
    'xl': SUB_ARCHITECTURE_BASE,
}

COMPATIBILITY_FAMILY_BY_ARCHITECTURE = {
    ARCHITECTURE_SD15: ARCHITECTURE_SD15,
    ARCHITECTURE_SDXL: ARCHITECTURE_SDXL,
}

RESOLUTION_SET_BY_ARCHITECTURE = {
    ARCHITECTURE_SD15: 'sd15',
    ARCHITECTURE_SDXL: 'sdxl',
}

PREFERRED_DEFAULT_ASPECT_RATIOS = {
    ARCHITECTURE_SD15: '768*512',
    ARCHITECTURE_SDXL: '1152*896',
}


@dataclass(frozen=True)
class ResolvedModelTaxonomy:
    architecture: str = DEFAULT_ARCHITECTURE
    sub_architecture: str | None = SUB_ARCHITECTURE_BASE
    compatibility_family: str | None = DEFAULT_COMPATIBILITY_FAMILY
    source: str = 'default'
    catalog_entry_id: str | None = None


def normalize_architecture(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return ARCHITECTURE_ALIASES.get(normalized, normalized)


def normalize_sub_architecture(value, architecture=None):
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if not normalized:
        return None

    normalized_architecture = normalize_architecture(architecture) if architecture is not None else None
    resolved = SUB_ARCHITECTURE_ALIASES.get(normalized, normalized)

    if resolved == SUB_ARCHITECTURE_NONE:
        return SUB_ARCHITECTURE_NONE

    if normalized_architecture is not None and normalized_architecture != ARCHITECTURE_SDXL:
        return SUB_ARCHITECTURE_BASE if resolved == SUB_ARCHITECTURE_BASE else None

    return resolved


def canonical_sub_architecture_for_path(root_key, architecture=None, sub_architecture=None):
    normalized_root_key = str(root_key or '').strip().lower()
    normalized_sub_architecture = normalize_sub_architecture(sub_architecture, architecture=architecture)

    if normalized_root_key in {'vae', 'embeddings'}:
        return SUB_ARCHITECTURE_NONE
    if normalized_sub_architecture in {None, SUB_ARCHITECTURE_NONE}:
        return SUB_ARCHITECTURE_BASE
    return normalized_sub_architecture


def build_canonical_relative_path(root_key, architecture, sub_architecture, name):
    normalized_architecture = normalize_architecture(architecture)
    if normalized_architecture is None:
        raise ValueError('Cannot derive relative_path without architecture.')

    normalized_name = os.path.basename(str(name or '').replace('\\', '/').strip())
    if not normalized_name:
        raise ValueError('Cannot derive relative_path without name.')

    path_sub_architecture = canonical_sub_architecture_for_path(
        root_key,
        architecture=normalized_architecture,
        sub_architecture=sub_architecture,
    )
    segments = [normalized_architecture]
    if path_sub_architecture not in {None, SUB_ARCHITECTURE_NONE}:
        segments.append(path_sub_architecture)
    segments.append(normalized_name)
    return '/'.join(segments)


def normalize_resolution_set_id(value):
    if value is None:
        return RESOLUTION_SET_BY_ARCHITECTURE[DEFAULT_ARCHITECTURE]
    normalized = str(value).strip().lower()
    if not normalized:
        return RESOLUTION_SET_BY_ARCHITECTURE[DEFAULT_ARCHITECTURE]
    return normalized


def resolve_resolution_set_id(architecture=None, sub_architecture=None):
    del sub_architecture
    normalized_architecture = normalize_architecture(architecture) or DEFAULT_ARCHITECTURE
    return RESOLUTION_SET_BY_ARCHITECTURE.get(
        normalized_architecture,
        RESOLUTION_SET_BY_ARCHITECTURE[DEFAULT_ARCHITECTURE],
    )


def get_preferred_aspect_ratio(architecture=None, sub_architecture=None):
    del sub_architecture
    normalized_architecture = normalize_architecture(architecture) or DEFAULT_ARCHITECTURE
    return PREFERRED_DEFAULT_ASPECT_RATIOS.get(
        normalized_architecture,
        PREFERRED_DEFAULT_ASPECT_RATIOS[DEFAULT_ARCHITECTURE],
    )


def get_compatibility_family(architecture=None, sub_architecture=None, model_type=None):
    del sub_architecture
    del model_type
    normalized_architecture = normalize_architecture(architecture) or DEFAULT_ARCHITECTURE
    return COMPATIBILITY_FAMILY_BY_ARCHITECTURE.get(
        normalized_architecture,
        DEFAULT_COMPATIBILITY_FAMILY,
    )


def build_resolved_model_taxonomy(
    architecture=None,
    sub_architecture=None,
    compatibility_family=None,
    source='default',
    catalog_entry_id=None,
):
    normalized_architecture = normalize_architecture(architecture) or DEFAULT_ARCHITECTURE
    normalized_sub_architecture = normalize_sub_architecture(sub_architecture, normalized_architecture)
    resolved_compatibility_family = compatibility_family or get_compatibility_family(
        architecture=normalized_architecture,
        sub_architecture=normalized_sub_architecture,
    )
    return ResolvedModelTaxonomy(
        architecture=normalized_architecture,
        sub_architecture=normalized_sub_architecture or SUB_ARCHITECTURE_BASE,
        compatibility_family=resolved_compatibility_family,
        source=source,
        catalog_entry_id=catalog_entry_id,
    )


def _normalize_path_segments(value):
    normalized = str(value or '').replace('\\', '/').strip().lower()
    if not normalized:
        return [], ''
    segments = [segment for segment in normalized.split('/') if segment not in {'', '.'}]
    basename = os.path.basename(normalized)
    return segments, basename


def infer_model_taxonomy_from_filename(filename):
    segments, basename = _normalize_path_segments(filename)
    if not basename:
        return None, None

    sub_architecture = None
    if SUB_ARCHITECTURE_PONY in segments or basename.startswith('pony_') or basename.startswith('pony-') or 'pony' in basename:
        sub_architecture = SUB_ARCHITECTURE_PONY
    elif SUB_ARCHITECTURE_ILLUSTRIOUS in segments or basename.startswith('il_') or basename.startswith('il-') or 'illustrious' in basename:
        sub_architecture = SUB_ARCHITECTURE_ILLUSTRIOUS
    elif SUB_ARCHITECTURE_NOOB in segments or basename.startswith('noob_') or basename.startswith('noob-') or 'noob' in basename:
        sub_architecture = SUB_ARCHITECTURE_NOOB

    architecture = None
    if ARCHITECTURE_SD15 in segments or 'sd1.5' in segments:
        architecture = ARCHITECTURE_SD15
    elif ARCHITECTURE_SDXL in segments or 'xl' in segments:
        architecture = ARCHITECTURE_SDXL
    elif basename.startswith('sd15_') or basename.startswith('sd15-') or 'sd1.5' in basename or 'sd15' in basename:
        architecture = ARCHITECTURE_SD15
    elif sub_architecture is not None:
        architecture = ARCHITECTURE_SDXL
    elif basename.startswith('xl_') or basename.startswith('xl-') or 'sdxl' in basename:
        architecture = ARCHITECTURE_SDXL

    if architecture != ARCHITECTURE_SDXL:
        sub_architecture = None

    return architecture, sub_architecture


def infer_architecture_from_filename(filename):
    architecture, _ = infer_model_taxonomy_from_filename(filename)
    return architecture
