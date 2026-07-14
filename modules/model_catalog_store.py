from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import modules.model_catalog_index as catalog_index
import modules.model_taxonomy as model_taxonomy
from modules.model_download.catalog import ModelCatalog
from modules.model_download.spec import (
    REGISTRATION_STATE_SOURCED_REGISTERED,
    ModelCatalogEntry,
)
from modules.model_manager_helpers import (
    LOCAL_REGISTERED_CATALOG_FILENAME,
    LOCAL_REGISTERED_CATALOG_ID,
    LOCAL_REGISTERED_CATALOG_LABEL,
    MANAGED_CATALOG_SCHEMA_VERSION,
    SYSTEM_CATALOG_IDS,
    USER_FACING_DISCOVERY_ROOT_KEYS,
    _default_display_name,
    _default_root_key_for_model_type,
    _extract_filename_from_url,
    _normalize_catalog_filename,
    _normalize_catalog_id,
    _normalize_catalog_label,
    _normalize_civitai_source_url,
    _normalize_generated_sub_architecture,
    _normalize_github_source_url,
    _normalize_huggingface_source_url,
    _normalize_lookup_key,
    _normalize_path,
    _normalize_source_provider_value,
    _slugify_identifier,
)


class ModelCatalogStore:
    def __init__(self, manager: Any):
        self._manager = manager

    def _catalog_path(self, source_path: str) -> Path:
        return Path(source_path).resolve()

    def _load_catalog_payload(self, source_path: str) -> dict[str, Any]:
        path = self._catalog_path(source_path)
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def _write_catalog_payload(self, source_path: str, payload: dict[str, Any]) -> None:
        path = self._catalog_path(source_path)
        path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + '\n', encoding='utf-8')

    def _normalize_catalog_source_path(self, source_path: str | os.PathLike[str]) -> str:
        return str(self._catalog_path(str(source_path)))

    def _is_writable_catalog_path(self, source_path: str | os.PathLike[str]) -> bool:
        writable_root = Path(self._manager._get_writable_catalog_directory()).resolve()
        try:
            Path(source_path).resolve().relative_to(writable_root)
        except ValueError:
            return False
        return True

    def _build_catalog_summary(self, source_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        entries = payload.get('entries', []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            entries = []
        catalog_id = str(payload.get('catalog_id') or '').strip() if isinstance(payload, dict) else ''
        return {
            'path': self._normalize_catalog_source_path(source_path),
            'file_name': Path(source_path).name,
            'catalog_id': catalog_id,
            'catalog_label': str(payload.get('catalog_label') or catalog_id).strip() if isinstance(payload, dict) else '',
            'source_provider': _normalize_source_provider_value(payload.get('source_provider') or 'local') if isinstance(payload, dict) else 'local',
            'entry_count': len([entry for entry in entries if isinstance(entry, dict)]),
            'writable': self._is_writable_catalog_path(source_path),
            'is_system_catalog': catalog_id in SYSTEM_CATALOG_IDS,
            'is_default_local_catalog': catalog_id == LOCAL_REGISTERED_CATALOG_ID,
        }

    def _iter_writable_catalog_payloads(self) -> list[tuple[str, dict[str, Any]]]:
        writable_dir = Path(self._manager._get_writable_catalog_directory())
        if not writable_dir.is_dir():
            return []
        results: list[tuple[str, dict[str, Any]]] = []
        for path in catalog_index.iter_catalog_files([writable_dir]):
            payload = self._load_catalog_payload(str(path))
            if not isinstance(payload, dict):
                continue
            results.append((self._normalize_catalog_source_path(str(path)), payload))
        return results

    def list_personal_catalogs(self, *, include_system: bool = False) -> list[dict[str, Any]]:
        catalogs: list[dict[str, Any]] = []
        for source_path, payload in self._iter_writable_catalog_payloads():
            summary = self._build_catalog_summary(source_path, payload)
            if summary['is_system_catalog'] and not include_system:
                continue
            catalogs.append(summary)
        catalogs.sort(key=lambda item: (item['is_system_catalog'], item['catalog_label'].lower(), item['catalog_id'].lower()))
        return catalogs

    def _validate_catalog_payload_for_write(
        self,
        payload: dict[str, Any],
        *,
        existing_source_path: str | None = None,
    ) -> tuple[dict[str, Any], list[ModelCatalogEntry]]:
        if not isinstance(payload, dict):
            raise ValueError('Catalog payload must be a JSON object.')

        normalized_payload = json.loads(json.dumps(payload))
        normalized_payload['schema_version'] = normalized_payload.get('schema_version') or MANAGED_CATALOG_SCHEMA_VERSION
        normalized_payload['catalog_id'] = _normalize_catalog_id(normalized_payload.get('catalog_id'))
        normalized_payload['catalog_label'] = _normalize_catalog_label(normalized_payload.get('catalog_label'))
        normalized_payload['source_provider'] = _normalize_source_provider_value(normalized_payload.get('source_provider') or 'local')
        if normalized_payload['catalog_id'] in SYSTEM_CATALOG_IDS:
            raise ValueError(f"Catalog id {normalized_payload['catalog_id']} is reserved.")

        entries = normalized_payload.get('entries', [])
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            raise ValueError('Catalog entries must be a list.')
        normalized_payload['entries'] = entries

        existing_path = self._normalize_catalog_source_path(existing_source_path) if existing_source_path else None
        for source in self._manager._ensure_catalog_index().list_sources():
            source_path = self._normalize_catalog_source_path(source.path)
            if existing_path and source_path == existing_path:
                continue
            if str(source.catalog_id or '').strip() == normalized_payload['catalog_id']:
                raise ValueError(f"Duplicate catalog id: {normalized_payload['catalog_id']}")

        parsed_catalog = ModelCatalog.from_dict(normalized_payload)
        parsed_entries = parsed_catalog.list()

        payload_entry_ids: set[str] = set()
        payload_aliases: set[str] = set()
        payload_relative_paths: set[tuple[str, str]] = set()
        for entry in parsed_entries:
            if entry.id in payload_entry_ids:
                raise ValueError(f'Duplicate entry id in catalog payload: {entry.id}')
            payload_entry_ids.add(entry.id)

            alias_key = str(entry.alias or '').strip().lower()
            if alias_key:
                if alias_key in payload_aliases:
                    raise ValueError(f'Duplicate alias in catalog payload: {entry.alias}')
                payload_aliases.add(alias_key)

            relative_key = _normalize_lookup_key(entry.relative_path)
            if relative_key:
                relative_tuple = (entry.root_key, relative_key)
                if relative_tuple in payload_relative_paths:
                    raise ValueError(f'Duplicate root_key + relative_path combination in catalog payload: {entry.root_key}:{entry.relative_path}')
                payload_relative_paths.add(relative_tuple)

        for record in self._manager._ensure_catalog_index().list_records():
            source_path = self._normalize_catalog_source_path(record.source_path)
            if existing_path and source_path == existing_path:
                continue
            if self._manager._is_auto_generated_unregistered_record(record):
                continue
            if record.entry.id in payload_entry_ids:
                raise ValueError(f'Duplicate entry id: {record.entry.id}')

            alias_key = str(record.entry.alias or '').strip().lower()
            if alias_key and alias_key in payload_aliases:
                raise ValueError(f'Duplicate alias: {record.entry.alias}')

            relative_key = _normalize_lookup_key(record.entry.relative_path)
            if relative_key and (record.entry.root_key, relative_key) in payload_relative_paths:
                raise ValueError(f'Duplicate root_key + relative_path combination: {record.entry.root_key}:{record.entry.relative_path}')

        return normalized_payload, parsed_entries

    def _build_managed_catalog_payload(
        self,
        *,
        catalog_id: str,
        catalog_label: str,
        source_provider: str,
        notes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'schema_version': MANAGED_CATALOG_SCHEMA_VERSION,
            'catalog_id': _normalize_catalog_id(catalog_id),
            'catalog_label': _normalize_catalog_label(catalog_label),
            'source_provider': _normalize_source_provider_value(source_provider),
            'entries': [],
        }
        normalized_notes = [str(note).strip() for note in (notes or []) if str(note).strip()]
        if normalized_notes:
            payload['notes'] = normalized_notes
        return payload

    def _resolve_personal_catalog_path(
        self,
        *,
        catalog_id: str | None = None,
        filename: str | None = None,
        default_source_provider: str = 'local',
    ) -> Path:
        normalized_catalog_id = _normalize_catalog_id(catalog_id) if catalog_id else None
        if normalized_catalog_id is not None:
            for metadata in self.list_personal_catalogs(include_system=True):
                if metadata['catalog_id'] == normalized_catalog_id:
                    return Path(metadata['path'])

        file_name = _normalize_catalog_filename(filename, default_stem=normalized_catalog_id or default_source_provider)
        return Path(self._manager._get_writable_catalog_directory()) / file_name

    def create_personal_catalog(
        self,
        *,
        catalog_id: str,
        catalog_label: str,
        source_provider: str,
        filename: str | None = None,
        notes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_managed_catalog_payload(
            catalog_id=catalog_id,
            catalog_label=catalog_label,
            source_provider=source_provider,
            notes=notes,
        )
        validated_payload, _ = self._validate_catalog_payload_for_write(payload)
        path = self._resolve_personal_catalog_path(
            catalog_id=validated_payload['catalog_id'],
            filename=filename,
            default_source_provider=validated_payload['source_provider'],
        )
        if path.exists():
            raise ValueError(f'Catalog file already exists: {path.name}')
        self._write_catalog_payload(str(path), validated_payload)
        self._manager.refresh_catalog_index(force_refresh=True)
        return self._build_catalog_summary(str(path), validated_payload)

    def import_personal_catalog(
        self,
        payload: dict[str, Any],
        *,
        filename: str | None = None,
    ) -> dict[str, Any]:
        validated_payload, _ = self._validate_catalog_payload_for_write(payload)
        path = self._resolve_personal_catalog_path(
            catalog_id=validated_payload['catalog_id'],
            filename=filename,
            default_source_provider=validated_payload['source_provider'],
        )
        if path.exists():
            raise ValueError(f'Catalog file already exists: {path.name}')
        self._write_catalog_payload(str(path), validated_payload)
        self._manager.refresh_catalog_index(force_refresh=True)
        return self._build_catalog_summary(str(path), validated_payload)

    def _default_personal_catalog_definition(self, source_provider: str) -> tuple[str, str, str]:
        del source_provider
        return 'user.personal.default', 'Personal Download Catalog', 'personal_download_catalog.catalog.json'

    def _ensure_default_personal_catalog(self, source_provider: str) -> dict[str, Any]:
        catalog_id, catalog_label, filename = self._default_personal_catalog_definition(source_provider)
        for metadata in self.list_personal_catalogs(include_system=True):
            if metadata['catalog_id'] == catalog_id:
                return metadata
        return self.create_personal_catalog(
            catalog_id=catalog_id,
            catalog_label=catalog_label,
            source_provider='local',
            filename=filename,
        )

    def add_sourced_model_entry(
        self,
        *,
        source_provider: str,
        source_input: str,
        model_type: str,
        architecture: str,
        sub_architecture: str | None = None,
        name: str | None = None,
        display_name: str | None = None,
        alias: str | None = None,
        relative_path: str | None = None,
        thumbnail_library_relative: str | None = None,
        asset_group_key: str | None = None,
        target_catalog_id: str | None = None,
        token_required: bool | None = None,
        token_env: str | None = None,
        source_version_id: str | None = None,
        entry_id: str | None = None,
    ) -> dict[str, Any]:
        provider = _normalize_source_provider_value(source_provider)
        if provider not in {'civitai', 'huggingface', 'github'}:
            raise ValueError('source_provider must be civitai, huggingface, or github for add-model flow.')

        normalized_model_type = str(model_type or '').strip().lower()
        root_key = _default_root_key_for_model_type(normalized_model_type)
        if root_key not in USER_FACING_DISCOVERY_ROOT_KEYS:
            raise ValueError(f'Unsupported model_type for add-model flow: {model_type}')

        normalized_architecture = model_taxonomy.normalize_architecture(architecture)
        if not normalized_architecture:
            raise ValueError('architecture is required')
        normalized_sub_architecture = _normalize_generated_sub_architecture(root_key, normalized_architecture, sub_architecture)
        compatibility_family = model_taxonomy.get_compatibility_family(
            architecture=normalized_architecture,
            sub_architecture=normalized_sub_architecture,
            model_type=normalized_model_type,
        )

        if provider == 'huggingface':
            normalized_source_version_id = str(source_version_id or '').strip() or None
            normalized_source_url = _normalize_huggingface_source_url(source_input)
            default_token_required = False
            default_token_env = 'HUGGINGFACE_TOKEN'
        elif provider == 'github':
            normalized_source_version_id = str(source_version_id or '').strip() or None
            normalized_source_url = _normalize_github_source_url(source_input)
            default_token_required = False
            default_token_env = None
        else:
            normalized_source_version_id, normalized_source_url = _normalize_civitai_source_url(source_input)
            default_token_required = True
            default_token_env = 'CIVITAI_TOKEN'

        normalized_name = str(name or '').strip() or _extract_filename_from_url(normalized_source_url)
        if not normalized_name:
            raise ValueError('name is required for add-model flow.')
        normalized_name = os.path.basename(normalized_name.replace('\\', '/').strip())
        if not normalized_name:
            raise ValueError('name is required for add-model flow.')

        normalized_display_name = str(display_name or '').strip() or _default_display_name(normalized_name)
        normalized_alias = str(alias or '').strip() or _slugify_identifier(normalized_display_name, default=_slugify_identifier(normalized_name))
        normalized_relative_path = _normalize_path(relative_path)
        if normalized_relative_path is None:
            normalized_relative_path = model_taxonomy.build_canonical_relative_path(root_key, normalized_architecture, normalized_sub_architecture, normalized_name)

        normalized_token_required = default_token_required if token_required is None else bool(token_required)
        normalized_token_env = str(token_env or '').strip() or (default_token_env if normalized_token_required else None)
        normalized_thumbnail_relative = _normalize_path(thumbnail_library_relative)
        stem_slug = _slugify_identifier(Path(normalized_name).stem, default='model')
        normalized_asset_group_key = str(asset_group_key).strip() if asset_group_key else f"{normalized_architecture}.{normalized_sub_architecture or 'general'}.{stem_slug}"
        normalized_entry_id = str(entry_id or '').strip() or '.'.join(filter(None, [provider, normalized_model_type, normalized_architecture, normalized_sub_architecture or 'general', stem_slug]))

        if target_catalog_id:
            target_metadata = None
            normalized_target_catalog_id = _normalize_catalog_id(target_catalog_id)
            for metadata in self.list_personal_catalogs(include_system=True):
                if metadata['catalog_id'] == normalized_target_catalog_id:
                    target_metadata = metadata
                    break
            if target_metadata is None:
                raise KeyError(f'Unknown personal catalog: {normalized_target_catalog_id}')
            if target_metadata['is_system_catalog']:
                raise ValueError(f'Catalog {normalized_target_catalog_id} is reserved for system use.')
            if target_metadata['source_provider'] not in {provider, 'local'}:
                raise ValueError(f"Catalog {normalized_target_catalog_id} is for {target_metadata['source_provider']} entries, not {provider}.")
            source_path = target_metadata['path']
            default_catalog_id = target_metadata['catalog_id']
            default_catalog_label = target_metadata['catalog_label']
        else:
            default_catalog = self._ensure_default_personal_catalog(provider)
            source_path = default_catalog['path']
            default_catalog_id = default_catalog['catalog_id']
            default_catalog_label = default_catalog['catalog_label']

        catalog_payload = self._load_personal_catalog_payload(
            source_path,
            default_catalog_id=default_catalog_id,
            default_catalog_label=default_catalog_label,
            source_provider=provider,
        )
        entries = [entry for entry in catalog_payload.get('entries', []) if isinstance(entry, dict)]
        entries.append({
            'id': normalized_entry_id,
            'name': normalized_name,
            'root_key': root_key,
            'relative_path': normalized_relative_path,
            'alias': normalized_alias,
            'display_name': normalized_display_name,
            'model_type': normalized_model_type,
            'architecture': normalized_architecture,
            'sub_architecture': normalized_sub_architecture,
            'compatibility_family': compatibility_family,
            'asset_group_key': normalized_asset_group_key,
            'thumbnail_library_relative': normalized_thumbnail_relative,
            'source_provider': provider,
            'source_version_id': normalized_source_version_id,
            'registration_state': REGISTRATION_STATE_SOURCED_REGISTERED,
            'visibility': 'generic',
            'preset_managed': False,
            'token_required': normalized_token_required,
            'source': {
                'url': normalized_source_url,
                'token_env': normalized_token_env,
            },
        })
        catalog_payload['entries'] = entries
        validated_payload, _ = self._validate_catalog_payload_for_write(catalog_payload, existing_source_path=source_path)
        self._write_catalog_payload(source_path, validated_payload)
        self._manager.refresh_catalog_index(force_refresh=True)
        entry = self._manager.get_entry(normalized_entry_id)
        if entry is None:
            raise RuntimeError(f'Catalog entry {normalized_entry_id} disappeared after write')
        return {
            'entry': entry,
            'catalog': self._build_catalog_summary(source_path, validated_payload),
        }

    def _iter_payload_entries(self, node: Any):
        if isinstance(node, list):
            for item in node:
                yield from self._iter_payload_entries(item)
        elif isinstance(node, dict):
            if 'id' in node and 'name' in node and 'root_key' in node:
                yield node
            else:
                for value in node.values():
                    yield from self._iter_payload_entries(value)

    def _local_registered_catalog_path(self) -> Path:
        path = self._resolve_personal_catalog_path(
            catalog_id=LOCAL_REGISTERED_CATALOG_ID,
            filename=LOCAL_REGISTERED_CATALOG_FILENAME,
            default_source_provider='local',
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_personal_catalog_payload(
        self,
        source_path: str | os.PathLike[str],
        *,
        default_catalog_id: str,
        default_catalog_label: str,
        source_provider: str,
    ) -> dict[str, Any]:
        path = Path(source_path)
        if not path.exists():
            return self._build_managed_catalog_payload(
                catalog_id=default_catalog_id,
                catalog_label=default_catalog_label,
                source_provider=source_provider,
            )
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault('schema_version', MANAGED_CATALOG_SCHEMA_VERSION)
        payload.setdefault('catalog_id', default_catalog_id)
        payload.setdefault('catalog_label', default_catalog_label)
        payload.setdefault('source_provider', source_provider)
        if not isinstance(payload.get('entries'), list):
            payload['entries'] = []
        return payload

    def _resolve_target_personal_catalog_source_path(
        self,
        *,
        target_catalog_id: str | None = None,
        existing_entry_id: str | None = None,
        source_provider: str = 'local',
    ) -> str:
        if existing_entry_id:
            existing_record = self._manager._find_catalog_record(existing_entry_id)
            if (
                existing_record is not None
                and self._is_writable_catalog_path(existing_record.source_path)
                and str(existing_record.catalog_id or '').strip() not in SYSTEM_CATALOG_IDS
            ):
                return self._normalize_catalog_source_path(existing_record.source_path)

        if target_catalog_id:
            normalized_catalog_id = _normalize_catalog_id(target_catalog_id)
            for metadata in self.list_personal_catalogs(include_system=True):
                if metadata['catalog_id'] != normalized_catalog_id:
                    continue
                if metadata['is_system_catalog']:
                    raise ValueError(f'Catalog {normalized_catalog_id} is reserved for system use.')
                return metadata['path']
            raise KeyError(f'Unknown personal catalog: {normalized_catalog_id}')

        del source_provider
        return self._normalize_catalog_source_path(self._local_registered_catalog_path())

    def _load_local_registered_catalog_payload(self) -> dict[str, Any]:
        return self._load_personal_catalog_payload(
            self._local_registered_catalog_path(),
            default_catalog_id=LOCAL_REGISTERED_CATALOG_ID,
            default_catalog_label=LOCAL_REGISTERED_CATALOG_LABEL,
            source_provider='local',
        )

    def _upsert_local_registered_entry(
        self,
        source_entry: ModelCatalogEntry,
        *,
        updates: dict[str, Any] | None = None,
        existing_entry_id: str | None = None,
        target_catalog_id: str | None = None,
    ) -> ModelCatalogEntry:
        payload = self._manager._prepare_registered_payload(source_entry, updates=updates)
        stable_relative = _normalize_path(payload.get('relative_path') or source_entry.relative_path) or source_entry.relative_path
        digest = hashlib.sha1(f"{source_entry.root_key}:{stable_relative}".encode('utf-8')).hexdigest()[:12]
        payload['id'] = existing_entry_id or f'user.local.{source_entry.root_key}.{digest}'

        payload_source_provider = _normalize_source_provider_value(payload.get('source_provider') or 'local')
        source_path = self._resolve_target_personal_catalog_source_path(
            target_catalog_id=target_catalog_id,
            existing_entry_id=existing_entry_id,
            source_provider=payload_source_provider,
        )
        catalog_payload = self._load_personal_catalog_payload(
            source_path,
            default_catalog_id=target_catalog_id or LOCAL_REGISTERED_CATALOG_ID,
            default_catalog_label=LOCAL_REGISTERED_CATALOG_LABEL,
            source_provider=payload_source_provider,
        )
        entries = [entry for entry in catalog_payload.get('entries', []) if isinstance(entry, dict)]
        updated = False
        for index, entry_data in enumerate(entries):
            if entry_data.get('id') != payload['id']:
                continue
            entries[index] = payload
            updated = True
            break
        if not updated:
            entries.append(payload)
        catalog_payload['entries'] = entries
        validated_payload, _ = self._validate_catalog_payload_for_write(
            catalog_payload,
            existing_source_path=source_path,
        )
        self._write_catalog_payload(source_path, validated_payload)
        self._manager.refresh_catalog_index(force_refresh=True)
        entry = self._manager.get_entry(payload['id'])
        if entry is None:
            raise RuntimeError(f"Local registered entry {payload['id']} disappeared after write")
        return entry

    def _replace_catalog_entry_payload(self, source_path: str, entry_id: str, entry_payload: dict[str, Any]) -> None:
        payload = self._load_catalog_payload(source_path)
        updated = False
        for existing_entry in self._iter_payload_entries(payload):
            if existing_entry.get('id') != entry_id:
                continue
            existing_entry.clear()
            existing_entry.update(entry_payload)
            updated = True
            break

        if not updated:
            payload.setdefault('entries', []).append(entry_payload)

        self._write_catalog_payload(source_path, payload)

    def _remove_catalog_entry_payload(self, source_path: str, entry_id: str) -> bool:
        payload = self._load_catalog_payload(source_path)
        entries = payload.get('entries', [])
        if not isinstance(entries, list):
            return False
        filtered_entries = [entry for entry in entries if not (isinstance(entry, dict) and entry.get('id') == entry_id)]
        if len(filtered_entries) == len(entries):
            return False
        payload['entries'] = filtered_entries
        self._write_catalog_payload(source_path, payload)
        return True
