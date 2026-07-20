from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

import modules.config as config
import modules.model_catalog_index as catalog_index
import modules.model_taxonomy as model_taxonomy
import modules.model_thumbnails as model_thumbnails
from modules.model_catalog_store import ModelCatalogStore
from modules.extra_utils import get_files_from_folder
from modules.model_download.spec import (
    REGISTRATION_STATE_LOCALLY_REGISTERED,
    REGISTRATION_STATE_SOURCED_REGISTERED,
    REGISTRATION_STATE_UNREGISTERED,
    ModelCatalogEntry,
)
from modules.model_manager_companions import ModelManagerCompanions
from modules.model_manager_helpers import (
    MODEL_FILE_EXTENSIONS,
    SYSTEM_CATALOG_IDS,
    _build_default_root_map,
    _default_filter_sub_architecture,
    _entry_to_payload,
    _matches_filter_scope,
    _normalize_path,
    _normalize_paths,
    _score_match_candidate,
)
from modules.model_manager_links import ModelManagerLinks
from modules.model_manager_runtime import (
    DownloadJobRegistry,
    DownloadJobState,
    InstalledModelLink,
    InstalledPathIndex,
    ModelInventoryRecord,
)

class ModelManager:

    def __init__(
        self,
        catalog_dirs: Iterable[str | os.PathLike[str]] | None = None,
        root_map: dict[str, Iterable[str | os.PathLike[str]]] | None = None,
        download_worker: Callable[[ModelCatalogEntry, Callable[[float | None, str | None], None]], Any] | None = None,
        writable_catalog_dir: str | os.PathLike[str] | None = None,
    ):
        self._catalog_dirs = [str(path) for path in catalog_dirs] if catalog_dirs is not None else None
        self._root_map = {
            key: _normalize_paths(paths)
            for key, paths in (root_map.items() if root_map is not None else _build_default_root_map().items())
        }
        self._catalog_index = None
        self._writable_catalog_dir = str(writable_catalog_dir) if writable_catalog_dir is not None else (
            self._catalog_dirs[-1] if self._catalog_dirs else None
        )
        self._installed_index: dict[str, InstalledPathIndex] = {}
        self._installed_links: list[InstalledModelLink] = []
        self._installed_links_by_entry_id: dict[str, list[InstalledModelLink]] = {}
        self._installed_links_by_relative_path: dict[str, list[InstalledModelLink]] = {}
        self._installed_links_by_name: dict[str, list[InstalledModelLink]] = {}
        self._installed_links_by_path: dict[str, InstalledModelLink] = {}
        self._index_lock = threading.RLock()
        self._download_worker = download_worker
        self.download_jobs = DownloadJobRegistry()
        self._catalog_store = ModelCatalogStore(self)
        self._links = ModelManagerLinks(self)
        self._companions = ModelManagerCompanions(self)

    @property
    def catalog_index(self):
        if self._catalog_index is None:
            self.refresh_catalog_index()
        return self._catalog_index

    @property
    def root_map(self) -> dict[str, list[str]]:
        return {key: list(paths) for key, paths in self._root_map.items()}

    def refresh_catalog_index(self, *, force_refresh: bool = False):
        catalog_dirs = self._catalog_dirs
        if catalog_dirs is None:
            catalog_dirs = config.get_model_catalog_directories()
        self._catalog_index = catalog_index.load_runtime_model_catalog_index(catalog_dirs, force_refresh=force_refresh)
        return self._catalog_index

    def refresh_installed_index(self):
        installed_index: dict[str, InstalledPathIndex] = {}
        for roots in self._root_map.values():
            for root in roots:
                if not os.path.isdir(root):
                    continue
                rel_to_abs: dict[str, str] = {}
                basenames: dict[str, str] = {}
                for relative_path in get_files_from_folder(root, extensions=MODEL_FILE_EXTENSIONS):
                    normalized_relative = _normalize_path(relative_path)
                    if normalized_relative is None:
                        continue
                    absolute_path = os.path.abspath(os.path.join(root, relative_path))
                    rel_to_abs.setdefault(normalized_relative, absolute_path)
                    basenames.setdefault(os.path.basename(relative_path).lower(), absolute_path)
                installed_index[root] = InstalledPathIndex(root_path=root, relative_paths=rel_to_abs, basenames=basenames)
        with self._index_lock:
            self._installed_index = installed_index
        return installed_index

    def refresh(self):
        self.refresh_catalog_index()
        self.refresh_installed_index()
        self.refresh_installed_links()
        self.sync_unregistered_install_catalog()
        return self

    def _ensure_catalog_index(self):
        if self._catalog_index is None:
            self.refresh_catalog_index()
        return self._catalog_index

    def _ensure_installed_index(self):
        with self._index_lock:
            if not self._installed_index:
                self._installed_index = self.refresh_installed_index()
            return dict(self._installed_index)

    def _installed_links_path(self) -> Path:
        return self._links._installed_links_path()

    def _load_installed_links_payload(self) -> dict[str, Any]:
        return self._links._load_installed_links_payload()

    def _write_installed_links_payload(self, payload: dict[str, Any]) -> None:
        self._links._write_installed_links_payload(payload)

    def refresh_installed_links(self) -> list[InstalledModelLink]:
        return self._links.refresh_installed_links()

    def _ensure_installed_links(self):
        return self._links._ensure_installed_links()

    def _find_installed_link_for_entry(self, entry_id: str) -> InstalledModelLink | None:
        return self._links._find_installed_link_for_entry(entry_id)

    def _ensure_persisted_installed_link(self, entry: ModelCatalogEntry, inventory_record: ModelInventoryRecord | None = None) -> InstalledModelLink | None:
        return self._links._ensure_persisted_installed_link(entry, inventory_record)

    def _find_installed_link_by_selector(self, selector: str, root_keys: Iterable[str] | None = None) -> InstalledModelLink | None:
        return self._links._find_installed_link_by_selector(selector, root_keys=root_keys)

    def _upsert_installed_link(
        self,
        *,
        entry_id: str,
        root_key: str,
        installed_path: str,
        installed_root_path: str | None = None,
        installed_relative_path: str | None = None,
    ) -> InstalledModelLink:
        return self._links._upsert_installed_link(
            entry_id=entry_id,
            root_key=root_key,
            installed_path=installed_path,
            installed_root_path=installed_root_path,
            installed_relative_path=installed_relative_path,
        )

    def _discovery_roots_for_key(self, root_key: str) -> list[str]:
        return self._links._discovery_roots_for_key(root_key)

    @staticmethod
    def _is_auto_generated_unregistered_record(record) -> bool:
        if record is None:
            return False
        tags = {str(tag) for tag in record.entry.tags}
        return (
            record.entry.registration_state == REGISTRATION_STATE_UNREGISTERED
            and 'auto_generated' in tags
        )

    def _preferred_record(self, records):
        records = list(records)
        if not records:
            return None
        for record in records:
            if not self._is_auto_generated_unregistered_record(record):
                return record
        return records[0]

    def _find_catalog_record(self, selector: str, root_keys: Iterable[str] | None = None):
        index = self._ensure_catalog_index()
        normalized = _normalize_path(selector)
        if normalized is None:
            return None

        relative_record = self._preferred_record(index.list_by_relative_path(normalized, root_keys=root_keys))
        if relative_record is not None and not self._is_auto_generated_unregistered_record(relative_record):
            return relative_record

        name_record = self._preferred_record(index.list_by_name(os.path.basename(normalized), root_keys=root_keys))
        if name_record is not None and not self._is_auto_generated_unregistered_record(name_record):
            return name_record

        if relative_record is not None:
            return relative_record
        if name_record is not None:
            return name_record

        record = index.get_record(normalized)
        if record is not None and (root_keys is None or record.entry.root_key in set(root_keys)):
            return record

        installed_link = self._find_installed_link_by_selector(normalized, root_keys=root_keys)
        if installed_link is not None:
            linked_record = index.get_record(installed_link.entry_id)
            if linked_record is not None:
                return linked_record

        return None

    def get_entry(self, selector: str, root_keys: Iterable[str] | None = None) -> ModelCatalogEntry | None:
        record = self._find_catalog_record(selector, root_keys=root_keys)
        return None if record is None else record.entry

    def _catalog_path(self, source_path: str) -> Path:
        return self._catalog_store._catalog_path(source_path)

    def _load_catalog_payload(self, source_path: str) -> dict[str, Any]:
        return self._catalog_store._load_catalog_payload(source_path)

    def _write_catalog_payload(self, source_path: str, payload: dict[str, Any]) -> None:
        self._catalog_store._write_catalog_payload(source_path, payload)

    def _normalize_catalog_source_path(self, source_path: str | os.PathLike[str]) -> str:
        return self._catalog_store._normalize_catalog_source_path(source_path)

    def _is_writable_catalog_path(self, source_path: str | os.PathLike[str]) -> bool:
        return self._catalog_store._is_writable_catalog_path(source_path)

    def _build_catalog_summary(self, source_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._catalog_store._build_catalog_summary(source_path, payload)

    def _iter_writable_catalog_payloads(self) -> list[tuple[str, dict[str, Any]]]:
        return self._catalog_store._iter_writable_catalog_payloads()

    def list_personal_catalogs(self, *, include_system: bool = False) -> list[dict[str, Any]]:
        return self._catalog_store.list_personal_catalogs(include_system=include_system)

    def _validate_catalog_payload_for_write(
        self,
        payload: dict[str, Any],
        *,
        existing_source_path: str | None = None,
    ) -> tuple[dict[str, Any], list[ModelCatalogEntry]]:
        return self._catalog_store._validate_catalog_payload_for_write(payload, existing_source_path=existing_source_path)

    def _build_managed_catalog_payload(
        self,
        *,
        catalog_id: str,
        catalog_label: str,
        source_provider: str,
        notes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return self._catalog_store._build_managed_catalog_payload(
            catalog_id=catalog_id,
            catalog_label=catalog_label,
            source_provider=source_provider,
            notes=notes,
        )

    def _resolve_personal_catalog_path(
        self,
        *,
        catalog_id: str | None = None,
        filename: str | None = None,
        default_source_provider: str = 'local',
    ) -> Path:
        return self._catalog_store._resolve_personal_catalog_path(
            catalog_id=catalog_id,
            filename=filename,
            default_source_provider=default_source_provider,
        )

    def create_personal_catalog(
        self,
        *,
        catalog_id: str,
        catalog_label: str,
        source_provider: str,
        filename: str | None = None,
        notes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return self._catalog_store.create_personal_catalog(
            catalog_id=catalog_id,
            catalog_label=catalog_label,
            source_provider=source_provider,
            filename=filename,
            notes=notes,
        )

    def import_personal_catalog(
        self,
        payload: dict[str, Any],
        *,
        filename: str | None = None,
    ) -> dict[str, Any]:
        return self._catalog_store.import_personal_catalog(payload, filename=filename)

    def _default_personal_catalog_definition(self, source_provider: str) -> tuple[str, str, str]:
        return self._catalog_store._default_personal_catalog_definition(source_provider)

    def _ensure_default_personal_catalog(self, source_provider: str) -> dict[str, Any]:
        return self._catalog_store._ensure_default_personal_catalog(source_provider)

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
        return self._catalog_store.add_sourced_model_entry(
            source_provider=source_provider,
            source_input=source_input,
            model_type=model_type,
            architecture=architecture,
            sub_architecture=sub_architecture,
            name=name,
            display_name=display_name,
            alias=alias,
            relative_path=relative_path,
            thumbnail_library_relative=thumbnail_library_relative,
            asset_group_key=asset_group_key,
            target_catalog_id=target_catalog_id,
            token_required=token_required,
            token_env=token_env,
            source_version_id=source_version_id,
            entry_id=entry_id,
        )

    def _iter_payload_entries(self, node: Any):
        yield from self._catalog_store._iter_payload_entries(node)


    def _get_writable_catalog_directory(self) -> str:
        return self._writable_catalog_dir or config.get_writable_model_catalog_directory()

    def _unregistered_catalog_path(self) -> Path:
        return self._links._unregistered_catalog_path()

    def _build_unregistered_catalog_payload(self, entries: list[ModelCatalogEntry]) -> dict[str, Any]:
        return self._links._build_unregistered_catalog_payload(entries)

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

    def suggest_catalog_matches(
        self,
        selector: str,
        *,
        limit: int = 3,
        source_provider: str | None = None,
        source_version_id: str | None = None,
    ) -> list[dict[str, Any]]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        return self._suggest_catalog_matches_for_entry(
            record.entry,
            limit=limit,
            source_provider=source_provider,
            source_version_id=source_version_id,
            exclude_entry_ids=[record.entry.id],
        )

    def _suggest_catalog_matches_for_entry(
        self,
        query_entry: ModelCatalogEntry,
        *,
        limit: int = 3,
        source_provider: str | None = None,
        source_version_id: str | None = None,
        exclude_entry_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        excluded = {str(value) for value in (exclude_entry_ids or []) if value}
        candidates: list[dict[str, Any]] = []
        for candidate_record in self._ensure_catalog_index().list_records():
            candidate_entry = candidate_record.entry
            if candidate_entry.id in excluded:
                continue
            if candidate_entry.root_key != query_entry.root_key:
                continue
            if self._is_auto_generated_unregistered_record(candidate_record):
                continue
            if candidate_entry.registration_state == REGISTRATION_STATE_UNREGISTERED:
                continue

            score, reasons = _score_match_candidate(
                query_entry,
                candidate_entry,
                source_provider=source_provider,
                source_version_id=source_version_id,
            )
            if score < 8.0 and 'version_id_exact' not in reasons:
                continue

            candidates.append({
                'score': round(score, 2),
                'reasons': reasons,
                'entry': self.inventory_record(candidate_entry).to_dict(),
            })

        candidates.sort(key=lambda item: (-item['score'], item['entry']['display_name'] or item['entry']['name']))
        return candidates[: max(1, int(limit))]

    def _find_companion_clip_catalog_entry(self, entry_like: ModelCatalogEntry | dict[str, Any] | None) -> ModelCatalogEntry | None:
        return self._companions._find_companion_clip_catalog_entry(entry_like)

    def _suggest_installed_companion_clips(
        self,
        *,
        target_clip_entry: ModelCatalogEntry | None,
        query_unet_entry: ModelCatalogEntry,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return self._companions._suggest_installed_companion_clips(
            target_clip_entry=target_clip_entry,
            query_unet_entry=query_unet_entry,
            limit=limit,
        )

    def resolve_companion_clip(self, selector_or_entry: str | ModelCatalogEntry, *, installed_only: bool = False) -> ModelCatalogEntry | None:
        return self._companions.resolve_companion_clip(selector_or_entry, installed_only=installed_only)

    def _build_unet_companion_clip_context(
        self,
        entry: ModelCatalogEntry,
        *,
        matched_selector: str | None = None,
        suggestions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._companions._build_unet_companion_clip_context(
            entry,
            matched_selector=matched_selector,
            suggestions=suggestions,
        )

    def build_registration_context(
        self,
        selector: str,
        *,
        suggest_limit: int = 3,
        source_provider: str | None = None,
        source_version_id: str | None = None,
        matched_selector: str | None = None,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        entry_record = self.inventory_record(record.entry)
        suggestions = self.suggest_catalog_matches(
            selector,
            limit=suggest_limit,
            source_provider=source_provider,
            source_version_id=source_version_id,
        )
        thumbnail = model_thumbnails.resolve_thumbnail(record.entry)
        context = {
            'entry': entry_record.to_dict(),
            'source_path': record.source_path,
            'suggestions': suggestions,
            'editable': True,
            'mode': 'registration',
            'personal_catalogs': self.list_personal_catalogs(),
            'thumbnail': {
                'relative_path': thumbnail.relative_path,
                'absolute_path': thumbnail.absolute_path,
                'exists': thumbnail.exists,
                'source': thumbnail.source,
            },
        }
        if record.entry.root_key == 'unet':
            context['companion_clip'] = self._build_unet_companion_clip_context(
                record.entry,
                matched_selector=matched_selector,
                suggestions=suggestions,
            )
        return context

    def build_installed_link_context(
        self,
        selector: str,
        *,
        suggest_limit: int = 3,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        inventory_record = self.inventory_record(record.entry)
        if not inventory_record.installed:
            raise ValueError('The selected model is not currently installed.')

        installed_link = self._find_installed_link_by_selector(selector, root_keys=[record.entry.root_key])
        if installed_link is None:
            installed_link = self._find_installed_link_for_entry(record.entry.id)
        if installed_link is None:
            installed_link = self._ensure_persisted_installed_link(record.entry, inventory_record)
        if installed_link is None:
            raise ValueError('The selected model does not have an editable installed link yet.')

        installed_relative_path = installed_link.installed_relative_path or inventory_record.installed_relative_path or record.entry.relative_path
        query_entry = ModelCatalogEntry(
            id=f'query.{record.entry.root_key}.{hashlib.sha1(str(installed_relative_path).encode("utf-8")).hexdigest()[:12]}',
            name=os.path.basename(installed_relative_path) or record.entry.name,
            root_key=record.entry.root_key,
            relative_path=installed_relative_path,
            display_name=inventory_record.entry.display_name,
            model_type=record.entry.model_type,
            architecture=record.entry.architecture,
            sub_architecture=record.entry.sub_architecture,
            compatibility_family=record.entry.compatibility_family,
            source_provider=record.entry.source_provider,
            source_version_id=record.entry.source_version_id,
            registration_state=record.entry.registration_state,
            visibility=record.entry.visibility,
            preset_managed=record.entry.preset_managed,
            token_required=record.entry.token_required,
            tags=record.entry.tags,
            asset_group_key=record.entry.asset_group_key,
            thumbnail_library_relative=record.entry.thumbnail_library_relative,
            source=record.entry.source,
        )
        suggestions = self._suggest_catalog_matches_for_entry(
            query_entry,
            limit=suggest_limit,
            source_provider=record.entry.source_provider,
            source_version_id=record.entry.source_version_id,
            exclude_entry_ids=[record.entry.id],
        )
        thumbnail = model_thumbnails.resolve_thumbnail(record.entry)
        return {
            'entry': inventory_record.to_dict(),
            'installed_link': installed_link.to_dict(),
            'source_path': record.source_path,
            'suggestions': suggestions,
            'editable': True,
            'mode': 'installed_link',
            'can_edit_catalog_fields': self._is_writable_catalog_path(record.source_path) and str(record.catalog_id or '').strip() not in SYSTEM_CATALOG_IDS,
            'personal_catalogs': self.list_personal_catalogs(),
            'thumbnail': {
                'relative_path': thumbnail.relative_path,
                'absolute_path': thumbnail.absolute_path,
                'exists': thumbnail.exists,
                'source': thumbnail.source,
            },
        }

    def _prepare_registered_payload(
        self,
        source_entry: ModelCatalogEntry,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = _entry_to_payload(source_entry)

        if matched_selector:
            matched_entry = self.get_entry(matched_selector)
            if matched_entry is None:
                raise KeyError(f'Unknown matched selector: {matched_selector}')
            if matched_entry.root_key != payload['root_key']:
                raise ValueError('Matched catalog entry must share the same root_key.')
            matched_payload = _entry_to_payload(matched_entry)
            for key in (
                'name',
                'display_name',
                'model_type',
                'architecture',
                'sub_architecture',
                'compatibility_family',
                'asset_group_key',
                'thumbnail_library_relative',
                'source_provider',
                'source_version_id',
                'visibility',
                'token_required',
            ):
                if matched_payload.get(key) is not None:
                    payload[key] = matched_payload[key]
            if matched_payload.get('source') is not None:
                payload['source'] = matched_payload['source']

        updates = dict(updates or {})
        if 'source_url' in updates:
            source_url = str(updates.pop('source_url') or '').strip()
            if source_url:
                payload['source'] = {'url': source_url}
            else:
                payload.pop('source', None)

        for key in (
            'alias',
            'display_name',
            'name',
            'relative_path',
            'model_type',
            'architecture',
            'sub_architecture',
            'compatibility_family',
            'asset_group_key',
            'thumbnail_library_relative',
            'source_provider',
            'source_version_id',
            'visibility',
        ):
            if key in updates and updates[key] is not None:
                payload[key] = updates[key]

        if 'token_env' in updates:
            token_env = updates['token_env']
            source_payload = dict(payload.get('source') or {})
            if source_payload or token_env:
                if token_env:
                    source_payload['token_env'] = str(token_env)
                payload['source'] = source_payload

        payload['architecture'] = model_taxonomy.normalize_architecture(payload.get('architecture')) or 'unknown'
        if payload.get('root_key') in {'vae', 'embeddings'} or payload.get('model_type') in {'vae', 'embedding'}:
            payload['sub_architecture'] = model_taxonomy.SUB_ARCHITECTURE_NONE
        else:
            normalized_sub_architecture = model_taxonomy.normalize_sub_architecture(
                payload.get('sub_architecture', 'general'),
                architecture=payload['architecture'],
            )
            payload['sub_architecture'] = normalized_sub_architecture or 'general'

        payload['compatibility_family'] = updates.get('compatibility_family') or model_taxonomy.get_compatibility_family(
            architecture=payload['architecture'],
            sub_architecture=payload['sub_architecture'],
            model_type=payload.get('model_type'),
        )

        provider = str(payload.get('source_provider') or 'local').strip().lower()
        source_payload = payload.get('source') if isinstance(payload.get('source'), dict) else None
        has_source_url = bool(source_payload and str(source_payload.get('url') or '').strip())
        payload['registration_state'] = (
            REGISTRATION_STATE_SOURCED_REGISTERED if has_source_url else REGISTRATION_STATE_LOCALLY_REGISTERED
        )
        if not has_source_url:
            payload.pop('source', None)
        if not provider:
            payload['source_provider'] = 'local'

        tags = [str(tag) for tag in payload.get('tags', []) if str(tag) not in {'auto_generated', 'unregistered'}]
        payload['tags'] = tags
        payload['preset_managed'] = False
        payload['token_required'] = bool(payload.get('token_required', False))
        return payload

    def _local_registered_catalog_path(self) -> Path:
        return self._catalog_store._local_registered_catalog_path()

    def _load_personal_catalog_payload(
        self,
        source_path: str | os.PathLike[str],
        *,
        default_catalog_id: str,
        default_catalog_label: str,
        source_provider: str,
    ) -> dict[str, Any]:
        return self._catalog_store._load_personal_catalog_payload(
            source_path,
            default_catalog_id=default_catalog_id,
            default_catalog_label=default_catalog_label,
            source_provider=source_provider,
        )

    def _resolve_target_personal_catalog_source_path(
        self,
        *,
        target_catalog_id: str | None = None,
        existing_entry_id: str | None = None,
        source_provider: str = 'local',
    ) -> str:
        return self._catalog_store._resolve_target_personal_catalog_source_path(
            target_catalog_id=target_catalog_id,
            existing_entry_id=existing_entry_id,
            source_provider=source_provider,
        )

    def _load_local_registered_catalog_payload(self) -> dict[str, Any]:
        return self._catalog_store._load_local_registered_catalog_payload()

    def _upsert_local_registered_entry(
        self,
        source_entry: ModelCatalogEntry,
        *,
        updates: dict[str, Any] | None = None,
        existing_entry_id: str | None = None,
        target_catalog_id: str | None = None,
    ) -> ModelCatalogEntry:
        return self._catalog_store._upsert_local_registered_entry(
            source_entry,
            updates=updates,
            existing_entry_id=existing_entry_id,
            target_catalog_id=target_catalog_id,
        )

    def _register_single_model_entry(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> ModelCatalogEntry:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')
        payload = self._prepare_registered_payload(record.entry, matched_selector=matched_selector, updates=updates)
        self._replace_catalog_entry_payload(record.source_path, record.entry.id, payload)
        self.refresh_catalog_index(force_refresh=True)
        refreshed = self.get_entry(record.entry.id)
        if refreshed is None:
            raise RuntimeError(f'Catalog entry {record.entry.id} disappeared after registration update')
        return refreshed

    def _register_unet_companion_clip(
        self,
        entry: ModelCatalogEntry,
        *,
        matched_selector: str | None = None,
        companion_selector: str | None = None,
        companion_relative_path: str | None = None,
    ) -> dict[str, Any] | None:
        return self._companions._register_unet_companion_clip(
            entry,
            matched_selector=matched_selector,
            companion_selector=companion_selector,
            companion_relative_path=companion_relative_path,
        )

    def register_model_entry_bundle(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
        target_catalog_id: str | None = None,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        inventory_record = self.inventory_record(record.entry)
        if not inventory_record.installed or not inventory_record.installed_path:
            raise ValueError('The selected model is not currently installed.')

        update_payload = dict(updates or {})
        existing_entry_id = record.entry.id if self._is_writable_catalog_path(record.source_path) and str(record.catalog_id or '').strip() not in SYSTEM_CATALOG_IDS else None

        if matched_selector:
            matched_entry = self.get_entry(matched_selector)
            if matched_entry is None:
                raise KeyError(f'Unknown matched selector: {matched_selector}')
            if matched_entry.root_key != record.entry.root_key:
                raise ValueError('Matched catalog entry must share the same root_key.')
            entry = matched_entry
        else:
            entry = self._upsert_local_registered_entry(
                record.entry,
                updates=update_payload,
                existing_entry_id=existing_entry_id,
                target_catalog_id=target_catalog_id,
            )

        installed_link = self._upsert_installed_link(
            entry_id=entry.id,
            root_key=record.entry.root_key,
            installed_path=inventory_record.installed_path,
            installed_root_path=inventory_record.installed_root_path,
            installed_relative_path=inventory_record.installed_relative_path or record.entry.relative_path,
        )

        if record.entry.registration_state == REGISTRATION_STATE_UNREGISTERED or self._is_auto_generated_unregistered_record(record):
            self._remove_catalog_entry_payload(record.source_path, record.entry.id)
            self.refresh_catalog_index(force_refresh=True)

        return {
            'entry': entry,
            'installed_link': installed_link.to_dict(),
        }

    def update_installed_model_link_bundle(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
        target_catalog_id: str | None = None,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        inventory_record = self.inventory_record(record.entry)
        if not inventory_record.installed:
            raise ValueError('The selected model is not currently installed.')

        current_link = self._find_installed_link_by_selector(selector, root_keys=[record.entry.root_key])
        if current_link is None:
            current_link = self._find_installed_link_for_entry(record.entry.id)
        if current_link is None and not inventory_record.installed_path:
            raise ValueError('The selected model does not have an editable installed link yet.')

        update_payload = dict(updates or {})
        installed_relative_path = _normalize_path(
            update_payload.pop('installed_relative_path', None)
            or (current_link.installed_relative_path if current_link is not None else None)
            or inventory_record.installed_relative_path
        )
        installed_root_path = inventory_record.installed_root_path or (current_link.installed_root_path if current_link is not None else None)
        installed_path = inventory_record.installed_path or (current_link.installed_path if current_link is not None else None)

        if installed_root_path and installed_relative_path:
            candidate_installed_path = os.path.abspath(os.path.join(installed_root_path, installed_relative_path))
            if not os.path.exists(candidate_installed_path):
                raise ValueError(f'Installed path not found: {installed_relative_path}')
            installed_path = candidate_installed_path

        if not installed_path:
            raise ValueError('Unable to resolve the installed path for this model.')

        catalog_updates = {
            key: value
            for key, value in update_payload.items()
            if key in {
                'alias',
                'display_name',
                'name',
                'relative_path',
                'model_type',
                'architecture',
                'sub_architecture',
                'compatibility_family',
                'asset_group_key',
                'thumbnail_library_relative',
                'source_provider',
                'source_version_id',
                'visibility',
                'source_url',
                'token_env',
            }
        }

        if matched_selector:
            target_entry = self.get_entry(matched_selector)
            if target_entry is None:
                raise KeyError(f'Unknown matched selector: {matched_selector}')
            if target_entry.root_key != record.entry.root_key:
                raise ValueError('Matched catalog entry must share the same root_key.')
        elif self._is_writable_catalog_path(record.source_path) and str(record.catalog_id or '').strip() not in SYSTEM_CATALOG_IDS:
            target_entry = self._upsert_local_registered_entry(
                record.entry,
                updates=catalog_updates,
                existing_entry_id=record.entry.id,
                target_catalog_id=target_catalog_id,
            ) if catalog_updates else record.entry
        elif catalog_updates:
            target_entry = self._upsert_local_registered_entry(
                record.entry,
                updates=catalog_updates,
                target_catalog_id=target_catalog_id,
            )
        else:
            target_entry = record.entry

        installed_link = self._upsert_installed_link(
            entry_id=target_entry.id,
            root_key=record.entry.root_key,
            installed_path=installed_path,
            installed_root_path=installed_root_path,
            installed_relative_path=installed_relative_path,
        )

        return {
            'entry': target_entry,
            'installed_link': installed_link.to_dict(),
        }

    def register_model_entry(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
        target_catalog_id: str | None = None,
    ) -> ModelCatalogEntry:
        return self.register_model_entry_bundle(
            selector,
            matched_selector=matched_selector,
            updates=updates,
            target_catalog_id=target_catalog_id,
        )['entry']

    def _build_unregistered_entry(self, root_key: str, relative_path: str) -> ModelCatalogEntry:
        return self._links._build_unregistered_entry(root_key, relative_path)

    def discover_unregistered_installed_entries(self) -> list[ModelCatalogEntry]:
        return self._links.discover_unregistered_installed_entries()

    def sync_unregistered_install_catalog(self) -> dict[str, Any]:
        return self._links.sync_unregistered_install_catalog()

    def update_catalog_entry_thumbnail_path(self, selector: str, thumbnail_library_relative: str) -> ModelCatalogEntry:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        normalized_relative = str(thumbnail_library_relative).replace('\\', '/').strip().lstrip('/')
        payload = self._load_catalog_payload(record.source_path)
        updated = False
        for entry_data in self._iter_payload_entries(payload):
            if entry_data.get('id') != record.entry.id:
                continue
            entry_data['thumbnail_library_relative'] = normalized_relative
            updated = True
            break

        if not updated:
            raise KeyError(f'Catalog entry {record.entry.id} not found in {record.source_path}')

        self._write_catalog_payload(record.source_path, payload)
        self.refresh_catalog_index(force_refresh=True)
        refreshed_entry = self.get_entry(record.entry.id)
        if refreshed_entry is None:
            raise RuntimeError(f'Catalog entry {record.entry.id} disappeared after catalog refresh')
        return refreshed_entry

    def persist_entry_thumbnail(
        self,
        selector: str,
        source: str | os.PathLike[str] | Any,
        *,
        slug: str | None = None,
        size: int | None = None,
    ) -> tuple[ModelCatalogEntry, model_thumbnails.ThumbnailResolution]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        resolution = model_thumbnails.persist_thumbnail_image(
            source,
            entry=record.entry,
            slug=slug,
            size=size,
        )
        refreshed_entry = self.update_catalog_entry_thumbnail_path(record.entry.id, resolution.relative_path)
        return refreshed_entry, resolution

    def resolve_entry_thumbnail(self, selector: str, *, slug: str | None = None) -> tuple[ModelCatalogEntry, model_thumbnails.ThumbnailResolution]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')
        return record.entry, model_thumbnails.resolve_thumbnail(record.entry, slug=slug)

    def _entry_installation(self, entry: ModelCatalogEntry) -> tuple[str | None, str | None, str | None]:
        installed_link = self._find_installed_link_for_entry(entry.id)
        if installed_link is not None and os.path.exists(installed_link.installed_path):
            return installed_link.installed_path, installed_link.installed_root_path, installed_link.installed_relative_path

        roots = self._root_map.get(entry.root_key, [])
        if not roots:
            return None, None, None

        canonical_relative = _normalize_path(entry.relative_path)
        candidate_names: list[str] = []
        for candidate in (entry.name, os.path.basename(entry.relative_path)):
            normalized = _normalize_path(candidate)
            if normalized and normalized not in candidate_names:
                candidate_names.append(normalized)

        installed_index = self._ensure_installed_index()
        for root in roots:
            root_index = installed_index.get(root)
            if root_index is None:
                continue
            if canonical_relative and canonical_relative in root_index.relative_paths:
                return root_index.relative_paths[canonical_relative], root, canonical_relative
            for candidate_name in candidate_names:
                matched = root_index.basenames.get(os.path.basename(candidate_name).lower())
                if matched is not None:
                    matched_relative = _normalize_path(os.path.relpath(matched, root).replace('\\', '/'))
                    return matched, root, matched_relative

        return None, None, None

    def inventory_record(self, entry: ModelCatalogEntry) -> ModelInventoryRecord:
        installed_path, installed_root, installed_relative = self._entry_installation(entry)
        return ModelInventoryRecord(
            entry=entry,
            installed=installed_path is not None,
            installed_path=installed_path,
            installed_root_path=installed_root,
            installed_relative_path=installed_relative,
        )

    def iter_inventory(
        self,
        *,
        architecture: str | None = None,
        sub_architecture: str | None = None,
        compatibility_family: str | None = None,
        model_type: str | None = None,
        root_key: str | None = None,
        registration_state: str | None = None,
        visibility: str | None = None,
        preset_managed: bool | None = None,
        installed: bool | None = None,
    ) -> list[ModelInventoryRecord]:
        entries = self._ensure_catalog_index().filter(
            architecture=architecture,
            sub_architecture=sub_architecture,
            compatibility_family=compatibility_family,
            model_type=model_type,
            root_key=root_key,
            registration_state=registration_state,
            visibility=visibility,
        )
        records = [self.inventory_record(entry) for entry in entries]
        if preset_managed is not None:
            records = [record for record in records if record.entry.preset_managed is preset_managed]
        if installed is not None:
            records = [record for record in records if record.installed is installed]
        return records

    def list_installed(self, **filters) -> list[ModelInventoryRecord]:
        return self.iter_inventory(installed=True, **filters)

    def list_available(self, **filters) -> list[ModelInventoryRecord]:
        return self.iter_inventory(installed=False, **filters)

    def list_dropdown_entries(
        self,
        *,
        base_model_name: str | None = None,
        root_key: str | None = None,
        installed_only: bool = True,
        generic_only: bool = True,
        include_preset_managed: bool = False,
        architecture: str | None = None,
        sub_architecture: str | None = None,
    ) -> list[ModelInventoryRecord]:
        if base_model_name is not None and architecture is None:
            scope = self.get_filter_scope(base_model_name, root_key=root_key, model_type='lora' if root_key == 'loras' else None)
            architecture = scope['architecture']
            sub_architecture = scope['sub_architecture']

        records = self.iter_inventory(
            architecture=architecture,
            sub_architecture=sub_architecture,
            root_key=root_key,
            installed=installed_only if installed_only else None,
        )
        if generic_only:
            records = [
                record
                for record in records
                if record.entry.visibility == 'generic' and (include_preset_managed or not record.entry.preset_managed)
            ]
        elif not include_preset_managed:
            records = [record for record in records if not record.entry.preset_managed]
        return records

    def list_installed_lora_dropdown_choices(
        self,
        *,
        base_model_name: str | None = None,
        include_preset_managed: bool = False,
    ) -> list[str]:
        scope = self.get_filter_scope(base_model_name, root_key='loras', model_type='lora')
        installed_index = self._ensure_installed_index()
        choices: list[str] = []
        seen: set[str] = set()

        for root_key in ('loras',):
            roots = self._root_map.get(root_key, [])
            for root in roots:
                root_index = installed_index.get(root)
                if root_index is None:
                    continue
                for relative_path in sorted(root_index.relative_paths):
                    normalized_relative_path = _normalize_path(relative_path)
                    if normalized_relative_path is None or normalized_relative_path in seen:
                        continue
                    entry = self.get_entry(normalized_relative_path, root_keys=[root_key])
                    if entry is not None:
                        if entry.model_type != 'lora':
                            continue
                        if entry.visibility != 'generic' and not include_preset_managed:
                            continue
                        if entry.preset_managed and not include_preset_managed:
                            continue
                        candidate_architecture = entry.architecture
                        candidate_sub_architecture = entry.sub_architecture
                    else:
                        taxonomy = config.resolve_model_taxonomy(
                            normalized_relative_path,
                            root_keys=(root_key,),
                            folder_paths=self._root_map.get(root_key, []),
                        )
                        candidate_architecture = taxonomy.architecture
                        candidate_sub_architecture = taxonomy.sub_architecture

                    if not _matches_filter_scope(
                        candidate_architecture,
                        candidate_sub_architecture,
                        target_architecture=scope['architecture'],
                        target_sub_architecture=scope['sub_architecture'],
                        root_key='loras',
                        model_type='lora',
                    ):
                        continue

                    seen.add(normalized_relative_path)
                    choices.append(normalized_relative_path)

        return sorted(choices)

    def build_architecture_groups(
        self,
        records: Iterable[ModelInventoryRecord] | None = None,
        **filters,
    ) -> list[dict[str, Any]]:
        entries = list(records) if records is not None else self.iter_inventory(**filters)
        buckets: dict[str, dict[str, Any]] = {}
        for record in entries:
            entry = record.entry
            bucket = buckets.setdefault(
                entry.architecture,
                {
                    'architecture': entry.architecture,
                    'compatibility_family': entry.compatibility_family,
                    'records': [],
                    'sub_architectures': {},
                    'installed_count': 0,
                    'available_count': 0,
                    'total_count': 0,
                },
            )
            bucket['records'].append(record.to_dict())
            bucket['total_count'] += 1
            if record.installed:
                bucket['installed_count'] += 1
            else:
                bucket['available_count'] += 1

            subgroup = bucket['sub_architectures'].setdefault(
                entry.sub_architecture,
                {
                    'sub_architecture': entry.sub_architecture,
                    'records': [],
                    'installed_count': 0,
                    'available_count': 0,
                    'total_count': 0,
                },
            )
            subgroup['records'].append(record.to_dict())
            subgroup['total_count'] += 1
            if record.installed:
                subgroup['installed_count'] += 1
            else:
                subgroup['available_count'] += 1
        return [buckets[key] for key in sorted(buckets)]

    def build_inventory_payload(self, **filters) -> dict[str, Any]:
        records = self.iter_inventory(**filters)
        return {
            'installed': [record.to_dict() for record in records if record.installed],
            'available': [record.to_dict() for record in records if not record.installed],
            'groups': self.build_architecture_groups(**filters),
        }

    def _resolve_taxonomy(self, selector: str):
        entry = self.get_entry(selector)
        if entry is not None:
            return model_taxonomy.build_resolved_model_taxonomy(
                architecture=entry.architecture,
                sub_architecture=entry.sub_architecture,
                compatibility_family=entry.compatibility_family,
                source='catalog',
                catalog_entry_id=entry.id,
            )
        try:
            return config.resolve_model_taxonomy(selector)
        except Exception:
            return model_taxonomy.build_resolved_model_taxonomy(source='default')

    def get_filter_scope(
        self,
        base_model_name: str | None,
        *,
        root_key: str | None = None,
        model_type: str | None = None,
    ) -> dict[str, Any]:
        if base_model_name is None:
            return {
                'architecture': None,
                'sub_architecture': None,
                'compatibility_family': None,
                'source': 'default',
                'catalog_entry_id': None,
            }

        taxonomy = self._resolve_taxonomy(base_model_name)
        return {
            'architecture': taxonomy.architecture,
            'sub_architecture': _default_filter_sub_architecture(
                taxonomy.architecture,
                taxonomy.sub_architecture,
                root_key=root_key,
                model_type=model_type,
            ),
            'compatibility_family': taxonomy.compatibility_family,
            'source': taxonomy.source,
            'catalog_entry_id': taxonomy.catalog_entry_id,
        }

    def start_download_job(
        self,
        selector: str,
        *,
        worker: Callable[[ModelCatalogEntry, Callable[[float | None, str | None], None]], Any] | None = None,
    ) -> DownloadJobState:
        entry = self.get_entry(selector)
        if entry is None:
            raise KeyError(f'Unknown model selector: {selector}')
        worker = worker or self._download_worker
        if worker is None:
            raise RuntimeError('No download worker has been configured')
        return self.download_jobs.submit(selector, entry, worker)

    def get_job(self, job_id: str) -> DownloadJobState | None:
        return self.download_jobs.get(job_id)

    def list_jobs(self) -> list[DownloadJobState]:
        return self.download_jobs.list_jobs()


default_model_manager = ModelManager()
