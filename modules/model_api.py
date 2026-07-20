from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import modules.model_taxonomy as model_taxonomy
from modules.model_download.policy import ModelDownloadPolicy
from modules.model_download.resolver import CivitAIResolver, DirectResolver, GitHubResolver, HuggingFaceResolver
from modules.model_download.transport import Aria2Transport
from modules.model_manager import ModelManager, default_model_manager


DEFAULT_DOWNLOAD_MESSAGE = 'Download queued'


def _resolve_browser_architecture_scope(
    *,
    root_key: str | None,
    base_model_name: str | None,
    architecture: str | None,
    sub_architecture: str | None,
) -> tuple[str | None, str | None]:
    if architecture is not None or base_model_name is not None:
        return architecture, sub_architecture

    if root_key in {'checkpoints', 'unet', 'loras', 'vae', 'embeddings'}:
        # W05 keeps dormant SD 1.5 records queryable via explicit filters, but
        # the live browser defaults should only surface active SDXL families.
        return model_taxonomy.ARCHITECTURE_SDXL, None

    return architecture, sub_architecture


def _queue_download_with_companion(manager: ModelManager, worker, entry, queued_jobs: list, skipped: list, queued_ids: set[str]):
    def try_queue(candidate_entry, *, reason_prefix: str = ''):
        if candidate_entry is None:
            return None
        if candidate_entry.id in queued_ids:
            skipped.append({'selector': candidate_entry.id, 'reason': f'{reason_prefix}already_queued'.strip('.')})
            return None
        inventory_record = manager.inventory_record(candidate_entry)
        if inventory_record.installed:
            skipped.append({'selector': candidate_entry.id, 'reason': f'{reason_prefix}already_installed'.strip('.')})
            return None
        if candidate_entry.registration_state == 'unregistered' or candidate_entry.source is None:
            skipped.append({'selector': candidate_entry.id, 'reason': f'{reason_prefix}not_downloadable'.strip('.')})
            return None
        try:
            job = manager.start_download_job(candidate_entry.id, worker=worker)
        except Exception as exc:
            skipped.append({'selector': candidate_entry.id, 'reason': f'{reason_prefix}{exc}'.strip('.')})
            return None
        queued_ids.add(candidate_entry.id)
        queued_jobs.append(job.to_dict())
        return job

    primary_job = try_queue(entry)
    if primary_job is None:
        return

    if entry.root_key == 'unet':
        companion_entry = manager.resolve_companion_clip(entry)
        if companion_entry is not None:
            try_queue(companion_entry, reason_prefix='companion_')



def _resolver_token_env(entry, default: str) -> str:
    if entry.source is not None and entry.source.token_env:
        return entry.source.token_env
    return default


def _select_resolver(entry):
    provider = (entry.source_provider or 'direct').lower()
    if provider == 'civitai':
        return CivitAIResolver(token_env=_resolver_token_env(entry, 'CIVITAI_TOKEN'))
    if provider in {'huggingface', 'hf'}:
        return HuggingFaceResolver(token_env=_resolver_token_env(entry, 'HUGGINGFACE_TOKEN'))
    if provider == 'github':
        return GitHubResolver()
    return DirectResolver()


def _execute_download(entry, manager: ModelManager, report_progress):
    policy = ModelDownloadPolicy(root_map=manager.root_map)
    resolver = _select_resolver(entry)
    report_progress(0.1, 'Resolving download plan')
    plan = resolver.resolve(entry, policy)
    report_progress(0.3, f'Downloading {entry.name}')
    result = Aria2Transport().download(plan)
    if result.success:
        manager.refresh_installed_index()
    return result


def _build_job_worker(manager: ModelManager, download_worker=None):
    if download_worker is None:
        def download_worker(entry, report_progress):
            return _execute_download(entry, manager, report_progress)

    def worker(entry, report_progress):
        result = download_worker(entry, report_progress)
        if getattr(result, 'success', False):
            manager.refresh_installed_index()
        return result

    return worker


def _serialize_records(records):
    return [record.to_dict() for record in records]


def _is_downloadable_browser_record(record) -> bool:
    entry = record.entry
    provider = str(entry.source_provider or '').strip().lower()
    has_source_url = bool(entry.source is not None and str(entry.source.url or '').strip())
    return has_source_url and provider in {'civitai', 'huggingface', 'hf', 'github'}


def _filter_ui_inventory_records(records):
    return list(records)


def _catalog_sources_payload(manager: ModelManager) -> list[dict[str, Any]]:
    sources = []
    for source in manager.catalog_index.list_sources():
        sources.append({
            'path': source.path,
            'catalog_id': source.catalog_id,
            'catalog_label': source.catalog_label,
            'entry_count': source.entry_count,
        })
    return sources


def _registration_updates_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get('updates'), dict):
        return dict(payload['updates'])

    ignored = {'selector', 'matched_selector', 'target_catalog_id'}
    return {key: value for key, value in payload.items() if key not in ignored}


def create_model_router(manager: ModelManager | None = None, download_worker=None) -> APIRouter:
    manager = manager or default_model_manager
    router = APIRouter()

    @router.get('/api/models/catalog')
    def list_catalog(
        architecture: str | None = Query(default=None),
        sub_architecture: str | None = Query(default=None),
        compatibility_family: str | None = Query(default=None),
        model_type: str | None = Query(default=None),
        root_key: str | None = Query(default=None),
        registration_state: str | None = Query(default=None),
        visibility: str | None = Query(default=None),
        preset_managed: bool | None = Query(default=None),
        installed: bool | None = Query(default=None),
    ):
        records = manager.iter_inventory(
            architecture=architecture,
            sub_architecture=sub_architecture,
            compatibility_family=compatibility_family,
            model_type=model_type,
            root_key=root_key,
            registration_state=registration_state,
            visibility=visibility,
            preset_managed=preset_managed,
            installed=installed,
        )
        records = _filter_ui_inventory_records(records)
        return JSONResponse(content={
            'entries': _serialize_records(records),
            'groups': manager.build_architecture_groups(records=records),
            'sources': _catalog_sources_payload(manager),
            'count': len(records),
        })


    @router.get('/api/models/personal-catalogs')
    def list_personal_catalogs(include_system: bool = Query(default=False)):
        catalogs = manager.list_personal_catalogs(include_system=include_system)
        return JSONResponse(content={
            'catalogs': catalogs,
            'count': len(catalogs),
        })

    @router.post('/api/models/personal-catalogs')
    def create_personal_catalog(payload: dict = Body(...)):
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail='Missing catalog payload')
        try:
            catalog = manager.create_personal_catalog(
                catalog_id=str(payload.get('catalog_id') or ''),
                catalog_label=str(payload.get('catalog_label') or ''),
                source_provider=str(payload.get('source_provider') or 'local'),
                filename=str(payload.get('filename') or '') or None,
                notes=payload.get('notes') if isinstance(payload.get('notes'), list) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content={
            'status': 'success',
            'catalog': catalog,
        })

    @router.post('/api/models/personal-catalogs/import')
    def import_personal_catalog(payload: dict = Body(...)):
        catalog_payload = payload.get('catalog') if isinstance(payload, dict) else None
        filename = payload.get('filename') if isinstance(payload, dict) else None
        if not isinstance(catalog_payload, dict):
            raise HTTPException(status_code=400, detail='Missing catalog')
        try:
            catalog = manager.import_personal_catalog(
                catalog_payload,
                filename=str(filename) if filename else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content={
            'status': 'success',
            'catalog': catalog,
        })

    @router.post('/api/models/add')
    def add_model(payload: dict = Body(...)):
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail='Missing add-model payload')
        try:
            result = manager.add_sourced_model_entry(
                source_provider=str(payload.get('source_provider') or ''),
                source_input=str(payload.get('source_input') or payload.get('source') or ''),
                model_type=str(payload.get('model_type') or ''),
                architecture=str(payload.get('architecture') or ''),
                sub_architecture=str(payload.get('sub_architecture')) if payload.get('sub_architecture') is not None else None,
                name=str(payload.get('name')) if payload.get('name') is not None else None,
                display_name=str(payload.get('display_name')) if payload.get('display_name') is not None else None,
                alias=str(payload.get('alias')) if payload.get('alias') is not None else None,
                relative_path=str(payload.get('relative_path')) if payload.get('relative_path') is not None else None,
                thumbnail_library_relative=str(payload.get('thumbnail_library_relative')) if payload.get('thumbnail_library_relative') is not None else None,
                asset_group_key=str(payload.get('asset_group_key')) if payload.get('asset_group_key') is not None else None,
                target_catalog_id=str(payload.get('target_catalog_id')) if payload.get('target_catalog_id') else None,
                token_required=payload.get('token_required') if isinstance(payload.get('token_required'), bool) else None,
                token_env=str(payload.get('token_env')) if payload.get('token_env') is not None else None,
                source_version_id=str(payload.get('source_version_id')) if payload.get('source_version_id') is not None else None,
                entry_id=str(payload.get('id')) if payload.get('id') is not None else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content={
            'status': 'success',
            'entry': manager.inventory_record(result['entry']).to_dict(),
            'catalog': result['catalog'],
        })

    @router.get('/api/models/installed')
    def list_installed(
        architecture: str | None = Query(default=None),
        sub_architecture: str | None = Query(default=None),
        compatibility_family: str | None = Query(default=None),
        model_type: str | None = Query(default=None),
        root_key: str | None = Query(default=None),
        registration_state: str | None = Query(default=None),
        preset_managed: bool | None = Query(default=None),
    ):
        records = manager.list_installed(
            architecture=architecture,
            sub_architecture=sub_architecture,
            compatibility_family=compatibility_family,
            model_type=model_type,
            root_key=root_key,
            registration_state=registration_state,
            preset_managed=preset_managed,
        )
        records = _filter_ui_inventory_records(records)
        return JSONResponse(content={
            'entries': _serialize_records(records),
            'groups': manager.build_architecture_groups(records=records),
            'count': len(records),
        })

    @router.get('/api/models/browser')
    def browser_payload(
        base_model_name: str | None = Query(default=None),
        root_key: str | None = Query(default=None),
        installed_only: bool = Query(default=False),
        generic_only: bool = Query(default=False),
        include_preset_managed: bool = Query(default=False),
        architecture: str | None = Query(default=None),
        sub_architecture: str | None = Query(default=None),
    ):
        scope = manager.get_filter_scope(base_model_name, root_key=root_key, model_type='lora' if root_key == 'loras' else None)
        if architecture is None:
            architecture = scope['architecture']
        if sub_architecture is None:
            sub_architecture = scope['sub_architecture']
        architecture, sub_architecture = _resolve_browser_architecture_scope(
            root_key=root_key,
            base_model_name=base_model_name,
            architecture=architecture,
            sub_architecture=sub_architecture,
        )

        records = manager.iter_inventory(
            architecture=architecture,
            sub_architecture=sub_architecture,
            root_key=root_key,
            installed=installed_only if installed_only else None,
            preset_managed=False if not include_preset_managed else None,
        )
        if generic_only:
            records = [record for record in records if record.entry.visibility == 'generic']
        if not include_preset_managed:
            records = [record for record in records if not record.entry.preset_managed]
        records = _filter_ui_inventory_records(records)

        installed_records = [record for record in records if record.installed]
        available_records = [record for record in records if not record.installed and _is_downloadable_browser_record(record)]
        installed_registered_records = [
            record for record in installed_records if record.entry.registration_state != 'unregistered'
        ]
        installed_unregistered_records = [
            record for record in installed_records if record.entry.registration_state == 'unregistered'
        ]
        available_registered_records = [
            record for record in available_records if record.entry.registration_state != 'unregistered'
        ]
        available_unregistered_records = [
            record for record in available_records if record.entry.registration_state == 'unregistered'
        ]
        return JSONResponse(content={
            'scope': scope,
            'installed': _serialize_records(installed_records),
            'available': _serialize_records(available_records),
            'installed_registered': _serialize_records(installed_registered_records),
            'installed_unregistered': _serialize_records(installed_unregistered_records),
            'available_registered': _serialize_records(available_registered_records),
            'available_unregistered': _serialize_records(available_unregistered_records),
            'groups': manager.build_architecture_groups(records=records),
            'count': len(records),
        })

    @router.get('/api/models/thumbnail')
    def get_thumbnail(selector: str = Query(...), slug: str | None = Query(default=None)):
        try:
            entry, resolution = manager.resolve_entry_thumbnail(selector, slug=slug)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content={
            'entry_id': entry.id,
            'selector': selector,
            'thumbnail': {
                'relative_path': resolution.relative_path,
                'absolute_path': resolution.absolute_path,
                'exists': resolution.exists,
                'source': resolution.source,
            },
        })


    @router.get('/api/models/thumbnail/file')
    def get_thumbnail_file(selector: str = Query(...), slug: str | None = Query(default=None)):
        try:
            _, resolution = manager.resolve_entry_thumbnail(selector, slug=slug)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not resolution.absolute_path:
            raise HTTPException(status_code=404, detail='Thumbnail not found')
        return FileResponse(resolution.absolute_path)

    @router.post('/api/models/thumbnail')
    def persist_thumbnail(payload: dict = Body(...)):
        selector = payload.get('selector') if isinstance(payload, dict) else None
        source_path = payload.get('source_path') if isinstance(payload, dict) else None
        slug = payload.get('slug') if isinstance(payload, dict) else None
        size = payload.get('size') if isinstance(payload, dict) else None

        if not selector:
            raise HTTPException(status_code=400, detail='Missing selector')
        if not source_path:
            raise HTTPException(status_code=400, detail='Missing source_path')

        try:
            entry, resolution = manager.persist_entry_thumbnail(
                str(selector),
                str(source_path),
                slug=slug,
                size=size,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        record = manager.inventory_record(entry)
        return JSONResponse(content={
            'status': 'success',
            'entry': record.to_dict(),
            'thumbnail': {
                'relative_path': resolution.relative_path,
                'absolute_path': resolution.absolute_path,
                'exists': resolution.exists,
                'source': resolution.source,
            },
        })

    @router.post('/api/models/thumbnail/upload')
    async def upload_thumbnail(
        selector: str = Form(...),
        file: UploadFile = File(...),
        slug: str | None = Form(default=None),
        size: int | None = Form(default=None),
    ):
        try:
            entry, resolution = manager.persist_entry_thumbnail(
                str(selector),
                file.file,
                slug=slug,
                size=size,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            await file.close()

        record = manager.inventory_record(entry)
        return JSONResponse(content={
            'status': 'success',
            'entry': record.to_dict(),
            'thumbnail': {
                'relative_path': resolution.relative_path,
                'absolute_path': resolution.absolute_path,
                'exists': resolution.exists,
                'source': resolution.source,
            },
        })

    @router.get('/api/models/registration')
    def registration_context(
        selector: str = Query(...),
        suggest_limit: int = Query(default=3),
        source_provider: str | None = Query(default=None),
        source_version_id: str | None = Query(default=None),
        matched_selector: str | None = Query(default=None),
    ):
        try:
            context = manager.build_registration_context(
                selector,
                suggest_limit=suggest_limit,
                source_provider=source_provider,
                source_version_id=source_version_id,
                matched_selector=matched_selector,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content=context)

    @router.post('/api/models/registration')
    def register_model(payload: dict = Body(...)):
        selector = payload.get('selector') if isinstance(payload, dict) else None
        matched_selector = payload.get('matched_selector') if isinstance(payload, dict) else None
        target_catalog_id = payload.get('target_catalog_id') if isinstance(payload, dict) else None
        if not selector:
            raise HTTPException(status_code=400, detail='Missing selector')

        try:
            result = manager.register_model_entry_bundle(
                str(selector),
                matched_selector=str(matched_selector) if matched_selector else None,
                updates=_registration_updates_from_payload(payload),
                target_catalog_id=str(target_catalog_id) if target_catalog_id else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response_payload = {
            'status': 'success',
            'entry': manager.inventory_record(result['entry']).to_dict(),
        }
        if result.get('installed_link') is not None:
            response_payload['installed_link'] = result['installed_link']
        if result.get('companion_clip') is not None:
            response_payload['companion_clip'] = result['companion_clip']
        return JSONResponse(content=response_payload)


    @router.get('/api/models/installed-link')
    def installed_link_context(
        selector: str = Query(...),
        suggest_limit: int = Query(default=3),
    ):
        try:
            context = manager.build_installed_link_context(
                selector,
                suggest_limit=suggest_limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content=context)

    @router.post('/api/models/installed-link')
    def update_installed_link(payload: dict = Body(...)):
        selector = payload.get('selector') if isinstance(payload, dict) else None
        matched_selector = payload.get('matched_selector') if isinstance(payload, dict) else None
        target_catalog_id = payload.get('target_catalog_id') if isinstance(payload, dict) else None
        if not selector:
            raise HTTPException(status_code=400, detail='Missing selector')

        try:
            result = manager.update_installed_model_link_bundle(
                str(selector),
                matched_selector=str(matched_selector) if matched_selector else None,
                updates=_registration_updates_from_payload(payload),
                target_catalog_id=str(target_catalog_id) if target_catalog_id else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response_payload = {
            'status': 'success',
            'entry': manager.inventory_record(result['entry']).to_dict(),
        }
        if result.get('installed_link') is not None:
            response_payload['installed_link'] = result['installed_link']
        return JSONResponse(content=response_payload)

    @router.post('/api/models/downloads/batch')
    def start_batch_download(payload: dict = Body(...)):
        selectors = payload.get('selectors') if isinstance(payload, dict) else None
        allowed_root_keys = []
        if isinstance(payload, dict):
            root_key = payload.get('root_key')
            root_keys = payload.get('root_keys')
            if root_key:
                allowed_root_keys.append(str(root_key))
            if isinstance(root_keys, list):
                allowed_root_keys.extend(str(value) for value in root_keys if value)
        allowed_root_keys = list(dict.fromkeys(allowed_root_keys))

        if not isinstance(selectors, list) or not selectors:
            raise HTTPException(status_code=400, detail='Missing selectors')
        if not allowed_root_keys:
            raise HTTPException(status_code=400, detail='Missing root_key')

        worker = _build_job_worker(manager, download_worker=download_worker)
        queued_jobs = []
        skipped = []
        queued_ids: set[str] = set()
        for selector in selectors:
            entry = manager.get_entry(str(selector))
            if entry is None:
                skipped.append({'selector': selector, 'reason': 'unknown'})
                continue
            if entry.root_key not in allowed_root_keys:
                skipped.append({'selector': selector, 'reason': 'different_root_key'})
                continue
            _queue_download_with_companion(manager, worker, entry, queued_jobs, skipped, queued_ids)

        status_code = 202 if queued_jobs else 200
        return JSONResponse(content={
            'status': 'queued' if queued_jobs else 'noop',
            'root_keys': allowed_root_keys,
            'queued_count': len(queued_jobs),
            'skipped_count': len(skipped),
            'jobs': queued_jobs,
            'skipped': skipped,
        }, status_code=status_code)

    @router.post('/api/models/download')
    def start_download(payload: dict = Body(...)):
        selector = payload.get('selector') if isinstance(payload, dict) else None
        if not selector and isinstance(payload, dict):
            selector = payload.get('model_id')
        if not selector:
            raise HTTPException(status_code=400, detail='Missing selector')

        worker = _build_job_worker(manager, download_worker=download_worker)
        entry = manager.get_entry(str(selector))
        if entry is None:
            raise HTTPException(status_code=404, detail=f'Unknown model selector: {selector}')
        queued_jobs = []
        skipped = []
        queued_ids: set[str] = set()
        try:
            _queue_download_with_companion(manager, worker, entry, queued_jobs, skipped, queued_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not queued_jobs:
            return JSONResponse(content={
                'status': 'noop',
                'message': DEFAULT_DOWNLOAD_MESSAGE,
                'jobs': [],
                'skipped': skipped,
            }, status_code=200)

        return JSONResponse(content={
            'status': 'queued',
            'message': DEFAULT_DOWNLOAD_MESSAGE,
            'job': queued_jobs[0],
            'jobs': queued_jobs,
            'skipped': skipped,
        }, status_code=202)

    @router.get('/api/models/downloads/{job_id}')
    def get_download_status(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Download job not found')
        return JSONResponse(content=job.to_dict())

    @router.post('/api/models/refresh')
    def refresh_models():
        manager.refresh()
        unregistered_records = manager.iter_inventory(registration_state='unregistered', installed=True)
        return JSONResponse(content={
            'status': 'success',
            'installed': len(manager.list_installed()),
            'available': len(manager.list_available()),
            'unregistered_installed': len(unregistered_records),
            'groups': manager.build_architecture_groups(),
        })

    return router


model_router = create_model_router(default_model_manager)


