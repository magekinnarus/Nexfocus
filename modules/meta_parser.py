import ast
import json
from abc import ABC, abstractmethod
from pathlib import Path
import os

from PIL import Image

import fooocus_version
import modules.config
from modules.flags import MetadataScheme
from modules.hash_cache import sha256_from_cache
from modules.util import get_file_from_folder_list, is_json

METADATA_APP_NAME = 'Nexfocus'


def _first_present(source: dict, *keys):
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _parse_list_value(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return []


def _decode_exif_text(value):
    if isinstance(value, bytes):
        for encoding in ('utf-8', 'utf-16', 'latin-1'):
            try:
                return value.decode(encoding).lstrip('\x00')
            except UnicodeDecodeError:
                continue
    return value


class MetadataParser(ABC):
    def __init__(self):
        self.raw_prompt: str = ''
        self.full_prompt: str = ''
        self.raw_negative_prompt: str = ''
        self.full_negative_prompt: str = ''
        self.steps: int = 30
        self.base_model_name: str = ''
        self.base_model_hash: str = ''
        self.loras: list = []
        self.vae_name: str = ''
        self.clip_model_name: str = ''

    @abstractmethod
    def get_scheme(self) -> MetadataScheme:
        raise NotImplementedError

    @abstractmethod
    def to_json(self, metadata: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def to_string(self, metadata: list) -> str:
        raise NotImplementedError

    def set_data(self, raw_prompt, full_prompt, raw_negative_prompt, full_negative_prompt, steps, base_model_name,
                 loras, vae_name, clip_model_name):
        self.raw_prompt = raw_prompt
        self.full_prompt = full_prompt
        self.raw_negative_prompt = raw_negative_prompt
        self.full_negative_prompt = full_negative_prompt
        self.steps = steps
        self.base_model_name = Path(base_model_name).stem

        base_model_path = get_file_from_folder_list(base_model_name, modules.config.paths_checkpoints)
        self.base_model_hash = sha256_from_cache(base_model_path)

        self.loras = []
        for (lora_name, lora_weight) in loras:
            if lora_name != 'None':
                lora_path = get_file_from_folder_list(lora_name, modules.config.paths_lora_lookup)
                lora_hash = sha256_from_cache(lora_path)
                self.loras.append((Path(lora_name).stem, lora_weight, lora_hash))
        self.vae_name = Path(vae_name).stem
        self.clip_model_name = Path(clip_model_name).stem if clip_model_name != 'None' else 'None'


class FooocusMetadataParser(MetadataParser):
    def __init__(self, scheme: MetadataScheme = MetadataScheme.FOOOCUS_NEX):
        super().__init__()
        self.scheme = scheme
        self.v2_record: dict | None = None

    def get_scheme(self) -> MetadataScheme:
        return self.scheme

    def set_v2_record(self, record: dict):
        self.v2_record = record

    def to_json(self, metadata: dict) -> dict:
        if isinstance(metadata, dict) and ('metadata_version' not in metadata or metadata.get('metadata_version') == 1):
            metadata = convert_v1_to_v2_metadata(metadata)

        if isinstance(metadata, dict):
            for key, value in list(metadata.items()):
                if value in ['', 'None', None]:
                    continue
                if key == 'base_model' and isinstance(value, str):
                    metadata[key] = self.replace_value_with_filename(key, value, modules.config.model_filenames) or value
                elif key == 'vae' and isinstance(value, str):
                    metadata[key] = self.replace_value_with_filename(key, value, modules.config.vae_filenames) or value
                elif key == 'clip_model' and isinstance(value, str):
                    metadata[key] = self.replace_value_with_filename(key, value, modules.config.clip_filenames) or value
                elif key == 'loras' and isinstance(value, list):
                    resolved_loras = []
                    for item in value:
                        if not isinstance(item, (list, tuple)) or len(item) < 2:
                            continue
                        name = str(item[0])
                        resolved_name = next(
                            (
                                filename
                                for filename in modules.config.lora_filenames
                                if Path(filename).stem == Path(name).stem
                            ),
                            name,
                        )
                        resolved_loras.append([resolved_name, item[1]])
                    metadata[key] = resolved_loras

        return metadata

    def to_string(self, metadata: list | dict) -> str:
        if self.v2_record is not None:
            return json.dumps(self.v2_record, indent=2)

        if isinstance(metadata, dict):
            if 'metadata_version' not in metadata:
                metadata = convert_v1_to_v2_metadata(metadata)
            return json.dumps(metadata, indent=2)

        if isinstance(metadata, list):
            res: dict = {k: v for _, k, v in metadata}
            res['full_prompt'] = self.full_prompt
            res['full_negative_prompt'] = self.full_negative_prompt
            res['steps'] = self.steps
            res['base_model'] = self.base_model_name
            res['base_model_hash'] = self.base_model_hash
            res['vae'] = self.vae_name
            res['clip_model'] = self.clip_model_name
            res['loras'] = self.loras
            return json.dumps(convert_v1_to_v2_metadata(res), indent=2)

        return json.dumps(metadata)

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        for filename in filenames:
            path = Path(filename)
            if key.startswith('lora_combined_'):
                if ' : ' in str(value):
                    name, weight = str(value).split(' : ', 1)
                    if name == path.stem:
                        return f'{filename} : {weight}'
            elif value == path.stem:
                return filename

        return None


def convert_v1_to_v2_metadata(v1_dict: dict) -> dict:
    if not isinstance(v1_dict, dict):
        return {}
    if v1_dict.get('metadata_version') == 2 and 'workflow' in v1_dict:
        return v1_dict

    workflow = 'txt2img'
    if 'inpaint_route' in v1_dict or 'Inpaint Route' in v1_dict or 'inpaint_prompt' in v1_dict:
        workflow = 'inpaint_sdxl'
    elif 'outpaint_prompt' in v1_dict:
        workflow = 'outpaint_sdxl'

    v2_record = {
        'metadata_version': 2,
        'workflow': workflow,
        'timestamp': str(_first_present(v1_dict, 'timestamp') or ''),
        'version': str(_first_present(v1_dict, 'version') or 'Fooocus Legacy v1'),
    }

    prompt_value = _first_present(v1_dict, 'prompt', 'Prompt')
    explicit_inpaint_prompt = _first_present(v1_dict, 'inpaint_prompt')
    if workflow == 'inpaint_sdxl' and explicit_inpaint_prompt is None and prompt_value is not None:
        # Legacy inpaint records stored only the effective merged prompt. Route
        # that best-effort value to the owning tab instead of duplicating it
        # into both the main and inpaint prompt controls.
        v2_record['inpaint_prompt'] = str(prompt_value)
    elif prompt_value is not None:
        v2_record['prompt'] = str(prompt_value)

    if explicit_inpaint_prompt is not None:
        v2_record['inpaint_prompt'] = str(explicit_inpaint_prompt)
    outpaint_prompt = _first_present(v1_dict, 'outpaint_prompt')
    if outpaint_prompt is not None:
        v2_record['outpaint_prompt'] = str(outpaint_prompt)

    simple_string_fields = {
        'negative_prompt': ('negative_prompt', 'Negative Prompt'),
        'base_model': ('base_model', 'Base Model'),
        'resolution': ('resolution', 'Resolution'),
        'sampler': ('sampler', 'Sampler'),
        'scheduler': ('scheduler', 'Scheduler'),
        'vae': ('vae', 'VAE'),
        'base_model_hash': ('base_model_hash',),
    }
    for target_key, source_keys in simple_string_fields.items():
        value = _first_present(v1_dict, *source_keys)
        if value is not None and str(value) not in ('', 'None'):
            v2_record[target_key] = str(value)

    styles_value = _first_present(v1_dict, 'styles', 'Styles')
    if styles_value is not None:
        v2_record['styles'] = _parse_list_value(styles_value)

    numeric_fields = {
        'steps': (int, ('steps', 'Steps')),
        'seed': (int, ('seed', 'Seed', 'task_seed')),
        'cfg_scale': (float, ('guidance_scale', 'Guidance Scale', 'cfg_scale')),
        'sharpness': (float, ('sharpness', 'Sharpness')),
        'clip_skip': (int, ('clip_skip', 'CLIP Skip')),
        'adaptive_cfg': (float, ('adaptive_cfg', 'CFG Mimicking from TSNR')),
    }
    for target_key, (cast_type, source_keys) in numeric_fields.items():
        value = _first_present(v1_dict, *source_keys)
        if value is None:
            continue
        try:
            v2_record[target_key] = cast_type(value)
        except (ValueError, TypeError):
            continue

    adm_guidance = _first_present(v1_dict, 'adm_guidance', 'ADM Guidance')
    if adm_guidance is not None:
        parsed_adm = _parse_list_value(adm_guidance)
        if len(parsed_adm) == 3:
            try:
                v2_record['adm_guidance'] = [float(value) for value in parsed_adm]
            except (ValueError, TypeError):
                pass

    loras = []
    raw_loras = v1_dict.get('loras')
    if isinstance(raw_loras, list):
        for item in raw_loras:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                loras.append([str(item[0]), float(item[1])])
            except (ValueError, TypeError):
                continue
    else:
        for i in range(1, 10):
            val = v1_dict.get(f'lora_combined_{i}') or v1_dict.get(f'LoRA {i}')
            if val and ' : ' in str(val):
                parts = str(val).split(' : ', 1)
                if len(parts) >= 2 and parts[0] != 'None':
                    try:
                        loras.append([parts[0], float(parts[1])])
                    except (ValueError, TypeError):
                        loras.append([parts[0], 1.0])
    if loras:
        v2_record['loras'] = loras

    if workflow == 'inpaint_sdxl':
        v2_record['inpaint_route'] = str(v1_dict.get('inpaint_route') or v1_dict.get('Inpaint Route') or 'sdxl')

    return v2_record


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    if metadata_scheme == MetadataScheme.FOOOCUS:
        return FooocusMetadataParser(MetadataScheme.FOOOCUS)
    if metadata_scheme == MetadataScheme.FOOOCUS_NEX:
        return FooocusMetadataParser(MetadataScheme.FOOOCUS_NEX)
    raise NotImplementedError


def read_info_from_image(file_or_path) -> tuple[dict | str | None, MetadataScheme | None]:
    if not file_or_path:
        return None, None

    try:
        if isinstance(file_or_path, (str, os.PathLike)):
            file_path = os.fspath(file_or_path)
            if not file_path or not os.path.exists(file_path):
                return None, None
            with Image.open(file_path) as img:
                items = (img.info or {}).copy()
                exif = img.getexif()
        else:
            items = (file_or_path.info or {}).copy()
            exif = file_or_path.getexif()
    except Exception:
        return None, None

    parameters = items.pop('parameters', None)
    metadata_scheme = items.pop('fooocus_scheme', None)
    if parameters is None and exif:
        parameters = _decode_exif_text(exif.get(0x9286))
    if metadata_scheme is None and exif:
        metadata_scheme = _decode_exif_text(exif.get(0x927C))
    if isinstance(metadata_scheme, str):
        try:
            metadata_scheme = MetadataScheme(metadata_scheme)
        except ValueError:
            metadata_scheme = None

    if parameters is not None:
        if is_json(parameters):
            parameters = json.loads(parameters)
        if isinstance(parameters, dict):
            if 'metadata_version' not in parameters or parameters.get('metadata_version') == 1:
                parameters = convert_v1_to_v2_metadata(parameters)
            if metadata_scheme is None:
                metadata_scheme = MetadataScheme.FOOOCUS_NEX

    return parameters, metadata_scheme


def get_exif(metadata: str | None, metadata_scheme: str):
    exif = Image.Exif()
    # tags see https://github.com/python-pillow/Pillow/blob/9.2.x/src/PIL/ExifTags.py
    # 0x9286 = UserComment
    exif[0x9286] = metadata
    # 0x0131 = Software
    exif[0x0131] = f'{METADATA_APP_NAME} {fooocus_version.version}'
    # 0x927C = MakerNote
    exif[0x927C] = metadata_scheme
    return exif




