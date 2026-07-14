from __future__ import annotations

import hashlib
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import modules.config as config
import modules.model_taxonomy as model_taxonomy
from modules.model_download.spec import ModelCatalogEntry
MODEL_FILE_EXTENSIONS = [
    '.pth',
    '.ckpt',
    '.bin',
    '.safetensors',
    '.fooocus.patch',
    '.gguf',
    '.sft',
]

USER_FACING_DISCOVERY_ROOT_KEYS = (
    'checkpoints',
    'loras',
    'unet',
    'clip',
    'vae',
    'embeddings',
)
UNREGISTERED_INSTALL_CATALOG_FILENAME = 'unregistered_install_catalog.catalog.json'
UNREGISTERED_INSTALL_CATALOG_ID = 'user.unregistered.install'
UNREGISTERED_INSTALL_CATALOG_LABEL = 'Unregistered Installed Models'
INSTALLED_MODEL_LINKS_FILENAME = 'installed_model_links.json'
INSTALLED_MODEL_LINKS_ID = 'user.installed.links'
INSTALLED_MODEL_LINKS_LABEL = 'Installed Model Links'
LOCAL_REGISTERED_CATALOG_FILENAME = 'user_local_models.catalog.json'
LOCAL_REGISTERED_CATALOG_ID = 'user.local.models'
LOCAL_REGISTERED_CATALOG_LABEL = 'User Local Models'
AUTO_GENERATED_UNREGISTERED_TAGS = ('auto_generated', 'unregistered')
MANAGED_CATALOG_SCHEMA_VERSION = 'm06-draft-1'
MANAGED_SOURCE_PROVIDERS = ('local', 'civitai', 'huggingface', 'github')
SYSTEM_CATALOG_IDS = {
    UNREGISTERED_INSTALL_CATALOG_ID,
    INSTALLED_MODEL_LINKS_ID,
}


def _normalize_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).replace('\\', '/').strip()
    return normalized or None


def _normalize_paths(values: Iterable[str | os.PathLike[str]] | str | os.PathLike[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    results: list[str] = []
    for value in values:
        normalized = os.path.abspath(os.path.realpath(str(value)))
        if normalized not in results:
            results.append(normalized)
    return results


def _normalize_absolute_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    normalized = os.path.abspath(os.path.realpath(str(value))).replace('\\', '/').strip()
    return normalized or None


def _normalize_lookup_key(value: str | os.PathLike[str] | None) -> str | None:
    normalized = _normalize_path(value)
    return None if normalized is None else normalized.lower()




def _normalize_catalog_id(value: str | None) -> str:
    normalized = str(value or '').strip()
    if not normalized:
        raise ValueError('catalog_id is required')
    return normalized


def _normalize_catalog_label(value: str | None) -> str:
    normalized = str(value or '').strip()
    if not normalized:
        raise ValueError('catalog_label is required')
    return normalized


def _normalize_source_provider_value(value: str | None, *, allow_empty: bool = False) -> str:
    normalized = str(value or '').strip().lower()
    if not normalized and allow_empty:
        return ''
    if normalized not in MANAGED_SOURCE_PROVIDERS:
        raise ValueError(
            f"source_provider must be one of {', '.join(MANAGED_SOURCE_PROVIDERS)}."
        )
    return normalized


def _normalize_catalog_filename(value: str | None, *, default_stem: str) -> str:
    raw = str(value or '').strip()
    if raw:
        raw = raw.replace('\\', '/').split('/')[-1]
        stem = raw
    else:
        stem = re.sub(r'[^a-z0-9._-]+', '_', default_stem.lower()).strip('._-') or 'catalog'

    lowered = stem.lower()
    if lowered.endswith('.catalog.json') or lowered.endswith('_catalog.json'):
        filename = stem
    elif lowered.endswith('.json'):
        filename = f'{stem[:-5]}.catalog.json'
    else:
        filename = f'{stem}.catalog.json'

    return re.sub(r'[^A-Za-z0-9._-]+', '_', filename)


def _slugify_identifier(value: str | None, *, default: str = 'entry') -> str:
    normalized = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')
    return normalized or default


def _default_display_name(value: str | None) -> str:
    stem = Path(str(value or '').strip()).stem
    display = stem.replace('_', ' ').strip()
    return display or 'Unnamed Model'


def _extract_filename_from_url(url: str) -> str | None:
    parsed = urlparse(str(url or '').strip())
    candidate = os.path.basename(parsed.path or '').strip()
    return candidate or None


def _extract_civitai_version_id(value: str | None) -> str | None:
    text = str(value or '').strip()
    if not text:
        return None
    if text.isdigit():
        return text

    parsed = urlparse(text)
    candidate = (parsed.path or '').strip('/')
    match = re.search(r'/api/download/models/(\d+)', parsed.path or '', re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'/api/v1/model-versions/(\d+)', parsed.path or '', re.IGNORECASE)
    if match:
        return match.group(1)
    query_values = parse_qs(parsed.query or '')
    for key in ('modelVersionId', 'modelversionid'):
        values = query_values.get(key)
        if values and str(values[0]).isdigit():
            return str(values[0])
    if candidate.isdigit():
        return candidate
    return None


def _normalize_huggingface_source_url(value: str | None) -> str:
    text = str(value or '').strip()
    if not text:
        raise ValueError('Hugging Face source URL is required.')
    parsed = urlparse(text)
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('Hugging Face source URL must start with http:// or https://.')
    host = str(parsed.netloc or '').strip().lower()
    if not host.endswith('huggingface.co'):
        raise ValueError('Hugging Face source URL must point to huggingface.co.')
    path = str(parsed.path or '').strip()
    if '/blob/' in path:
        path = path.replace('/blob/', '/resolve/', 1)
    if not os.path.basename(path):
        raise ValueError('Hugging Face source URL must point to a specific file.')
    return parsed._replace(scheme='https', path=path, query='', fragment='').geturl()


def _normalize_github_source_url(value: str | None) -> str:
    text = str(value or '').strip()
    if not text:
        raise ValueError('GitHub source URL is required.')
    parsed = urlparse(text)
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('GitHub source URL must start with http:// or https://.')
    host = str(parsed.netloc or '').strip().lower()
    if host not in {'github.com', 'raw.githubusercontent.com'} and not host.endswith('.githubusercontent.com'):
        raise ValueError('GitHub source URL must point to github.com or githubusercontent.com.')
    if not os.path.basename(parsed.path or ''):
        raise ValueError('GitHub source URL must point to a specific file.')
    return parsed._replace(scheme='https', query='', fragment='').geturl()


def _normalize_civitai_source_url(value: str | None) -> tuple[str, str]:
    version_id = _extract_civitai_version_id(value)
    if version_id is None:
        raise ValueError('CivitAI source must be a numeric model version id or a valid CivitAI link.')
    return version_id, f'https://civitai.com/api/download/models/{version_id}'
def _build_default_root_map() -> dict[str, list[str]]:
    root_map: dict[str, list[str]] = {
        'checkpoints': _normalize_paths(config.paths_checkpoints),
        'loras': _normalize_paths(config.paths_loras),
        'loras_lcm': _normalize_paths(config.path_loras_lcm),
        'loras_lightning': _normalize_paths(config.path_loras_lightning),
        'embeddings': _normalize_paths(config.path_embeddings),
        'vae_approx': _normalize_paths(config.path_vae_approx),
        'vae': _normalize_paths(config.path_vae),
        'unet': _normalize_paths(config.path_unet),
        'clip': _normalize_paths(config.paths_clips),
        'upscale_models': _normalize_paths(config.path_upscale_models),
        'inpaint': _normalize_paths(config.path_inpaint),
        'controlnet_models': _normalize_paths(config.path_controlnet),
        'clip_vision': _normalize_paths(config.path_clip_vision),
        'vision_support': _normalize_paths(config.path_vision_support),
        'preprocessors': _normalize_paths(config.path_preprocessors),
        'insightface': _normalize_paths(config.path_insightface),
        'removals': _normalize_paths(config.path_removals),
    }
    return {key: paths for key, paths in root_map.items() if paths}

def _coerce_result_path(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result
    for attr in ('destination_path', 'path', 'result_path'):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(result, dict):
        for key in ('destination_path', 'path', 'result_path'):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _coerce_result_message(result: Any) -> str:
    if result is None:
        return ''
    if isinstance(result, str):
        return result
    message = getattr(result, 'message', None)
    if isinstance(message, str) and message:
        return message
    if isinstance(result, dict):
        value = result.get('message')
        if isinstance(value, str) and value:
            return value
    return ''

def _coerce_result_success(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, dict) and 'success' in result:
        return bool(result.get('success'))
    success = getattr(result, 'success', None)
    if success is None:
        return True
    return bool(success)


def _default_filter_sub_architecture(
    architecture: str | None,
    sub_architecture: str | None,
    *,
    root_key: str | None = None,
    model_type: str | None = None,
) -> str | None:
    architecture = model_taxonomy.normalize_architecture(architecture)
    sub_architecture = model_taxonomy.normalize_sub_architecture(sub_architecture, architecture=architecture)

    if architecture != model_taxonomy.ARCHITECTURE_SDXL:
        return None

    if root_key == 'loras' and model_type == 'lora' and sub_architecture == model_taxonomy.SUB_ARCHITECTURE_NOOB:
        return model_taxonomy.SUB_ARCHITECTURE_ILLUSTRIOUS

    return sub_architecture



def _default_model_type_for_root(root_key: str) -> str:
    return {
        'checkpoints': 'checkpoint',
        'loras': 'lora',
        'unet': 'unet',
        'clip': 'clip',
        'vae': 'vae',
        'embeddings': 'embedding',
    }.get(root_key, root_key)


def _default_root_key_for_model_type(model_type: str) -> str:
    normalized = str(model_type or '').strip().lower()
    return {
        'checkpoint': 'checkpoints',
        'checkpoints': 'checkpoints',
        'lora': 'loras',
        'loras': 'loras',
        'unet': 'unet',
        'clip': 'clip',
        'vae': 'vae',
        'embedding': 'embeddings',
        'embeddings': 'embeddings',
    }.get(normalized, normalized)


def _normalize_generated_sub_architecture(root_key: str, architecture: str | None, sub_architecture: str | None) -> str | None:
    if root_key in {'vae', 'embeddings'}:
        return model_taxonomy.SUB_ARCHITECTURE_NONE

    filtered = _default_filter_sub_architecture(
        architecture,
        sub_architecture,
        root_key=root_key,
        model_type=_default_model_type_for_root(root_key),
    )
    if filtered is not None:
        return filtered

    if model_taxonomy.normalize_architecture(architecture) in {model_taxonomy.ARCHITECTURE_SD15, model_taxonomy.ARCHITECTURE_SDXL}:
        return model_taxonomy.SUB_ARCHITECTURE_BASE
    return None


def _build_unregistered_entry_id(root_key: str, relative_path: str) -> str:
    digest = hashlib.sha1(f'{root_key}:{relative_path}'.encode('utf-8')).hexdigest()[:12]
    return f'unregistered.{root_key}.{digest}'


def _normalize_match_name(value: str | None) -> str:
    if value is None:
        return ''
    normalized = Path(str(value)).stem.lower()
    normalized = re.sub(r'[\W_]+', ' ', normalized)
    normalized = re.sub(r'\b(sd15|sd 15|sdxl|xl|checkpoint|model|lora|vae|clip|clips|embedding|embeddings|gguf)\b', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _tokenize_match_name(value: str | None) -> set[str]:
    normalized = _normalize_match_name(value)
    if not normalized:
        return set()
    return {token for token in normalized.split(' ') if token}


def _score_match_candidate(
    query_entry: ModelCatalogEntry,
    candidate: ModelCatalogEntry,
    *,
    source_provider: str | None = None,
    source_version_id: str | None = None,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    query_tokens = _tokenize_match_name(query_entry.name)
    candidate_labels = [candidate.name, candidate.display_name, candidate.alias]
    candidate_norms = [_normalize_match_name(label) for label in candidate_labels if label]
    candidate_tokens = set()
    for label in candidate_labels:
        candidate_tokens.update(_tokenize_match_name(label))

    if source_provider and str(candidate.source_provider).lower() == str(source_provider).lower():
        score += 8.0
        reasons.append(f'provider:{candidate.source_provider}')

    if source_version_id and candidate.source_version_id and str(candidate.source_version_id) == str(source_version_id):
        score += 100.0
        reasons.append('version_id_exact')

    query_norm = _normalize_match_name(query_entry.name)
    best_ratio = 0.0
    exact_name = False
    for candidate_norm in candidate_norms:
        if not candidate_norm:
            continue
        if candidate_norm == query_norm and query_norm:
            exact_name = True
        best_ratio = max(best_ratio, SequenceMatcher(None, query_norm, candidate_norm).ratio())
    if exact_name:
        score += 35.0
        reasons.append('name_exact')
    if best_ratio > 0.0:
        score += best_ratio * 40.0
        reasons.append(f'name_similarity:{best_ratio:.2f}')

    if query_tokens and candidate_tokens:
        overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), len(candidate_tokens), 1)
        if overlap > 0:
            score += overlap * 25.0
            reasons.append(f'token_overlap:{overlap:.2f}')

    if query_entry.architecture and query_entry.architecture == candidate.architecture:
        score += 6.0
        reasons.append(f'architecture:{candidate.architecture}')
    if query_entry.sub_architecture and query_entry.sub_architecture == candidate.sub_architecture:
        score += 3.0
        reasons.append(f'sub_architecture:{candidate.sub_architecture}')

    return score, reasons




_UNET_COMPANION_QUANT_SUFFIX_RE = re.compile(r'_(q\d(?:_[a-z0-9]+)*)$', re.IGNORECASE)


def _derive_clip_name_from_unet_name(value: str | None) -> str:
    filename = os.path.basename(str(value or '').strip())
    if not filename:
        return 'paired_clips.safetensors'
    stem, _ = os.path.splitext(filename)
    cleaned = _UNET_COMPANION_QUANT_SUFFIX_RE.sub('', stem).strip(' _-')
    if not cleaned:
        cleaned = stem or 'paired'
    return f'{cleaned}_clips.safetensors'


def _derive_clip_display_name_from_unet_payload(payload: dict[str, Any]) -> str:
    source = str(payload.get('display_name') or payload.get('name') or '').strip()
    if not source:
        return 'paired clips'
    stem = Path(source).stem if os.path.splitext(source)[1] else source
    stem = re.sub(r'\bq\d(?:\s+[a-z0-9]+)*\b$', '', stem, flags=re.IGNORECASE).strip(' _-')
    stem = re.sub(r'[_\-]+', ' ', stem).strip()
    return f'{stem or "paired"} clips'

def _entry_to_payload(entry: ModelCatalogEntry) -> dict[str, Any]:
    payload = {
        'id': entry.id,
        'name': entry.name,
        'root_key': entry.root_key,
        'relative_path': entry.relative_path,
        'display_name': entry.display_name,
        'model_type': entry.model_type,
        'architecture': entry.architecture,
        'sub_architecture': entry.sub_architecture,
        'compatibility_family': entry.compatibility_family,
        'source_provider': entry.source_provider,
        'registration_state': entry.registration_state,
        'visibility': entry.visibility,
        'preset_managed': entry.preset_managed,
        'token_required': entry.token_required,
        'tags': list(entry.tags),
    }
    if entry.alias is not None:
        payload['alias'] = entry.alias
    if entry.asset_group_key is not None:
        payload['asset_group_key'] = entry.asset_group_key
    if entry.thumbnail_library_relative is not None:
        payload['thumbnail_library_relative'] = entry.thumbnail_library_relative
    if entry.source_version_id is not None:
        payload['source_version_id'] = entry.source_version_id
    if entry.source is not None:
        payload['source'] = {
            'url': entry.source.url,
            'token_env': entry.source.token_env,
            'headers': [list(header) for header in entry.source.headers],
        }
    return payload


def _matches_filter_scope(
    candidate_architecture: str | None,
    candidate_sub_architecture: str | None,
    *,
    target_architecture: str | None = None,
    target_sub_architecture: str | None = None,
    root_key: str | None = None,
    model_type: str | None = None,
) -> bool:
    normalized_target_architecture = model_taxonomy.normalize_architecture(target_architecture)
    normalized_target_sub_architecture = _default_filter_sub_architecture(
        normalized_target_architecture,
        target_sub_architecture,
        root_key=root_key,
        model_type=model_type,
    )
    normalized_candidate_architecture = model_taxonomy.normalize_architecture(candidate_architecture)
    normalized_candidate_sub_architecture = _default_filter_sub_architecture(
        normalized_candidate_architecture,
        candidate_sub_architecture,
        root_key=root_key,
        model_type=model_type,
    )

    if normalized_target_architecture is not None and normalized_candidate_architecture != normalized_target_architecture:
        return False
    if root_key == 'loras' or model_type == 'lora':
        return True
    if normalized_target_sub_architecture is not None and normalized_candidate_sub_architecture != normalized_target_sub_architecture:
        return False
    return True


