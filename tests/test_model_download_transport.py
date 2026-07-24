import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import types

mock_args = types.SimpleNamespace()
mock_args.colab = False
mock_args.preset = None
mock_args.output_path = None
mock_args.temp_path = None
mock_args.skip_model_load = False

sys.modules['args_manager'] = MagicMock()
sys.modules['args_manager'].args = mock_args

from modules.model_download.orchestrator import ModelDownloadOrchestrator
from modules.model_download.policy import ModelDownloadPolicy
from modules.model_download.resolver import CivitAIResolver, GitHubResolver, HuggingFaceResolver
from modules.model_download.runtime import download_file
from modules.model_download.spec import DownloadPlan, ModelCatalogEntry, ModelSource
from modules.model_download.transport import Aria2Transport, FallbackTransport


def _entry() -> ModelCatalogEntry:
    return ModelCatalogEntry(
        id='entry.test',
        name='test_model.safetensors',
        root_key='checkpoints',
        relative_path='sdxl/base/test_model.safetensors',
        architecture='sdxl',
        sub_architecture='base',
        compatibility_family='sdxl',
        source=ModelSource(url='https://example.com/model.safetensors'),
    )


def test_aria2_transport_downloads_to_nested_destination(tmp_path, monkeypatch):
    entry = _entry()
    destination_root = tmp_path / 'checkpoints'
    plan = DownloadPlan(
        entry=entry,
        destination_root=str(destination_root),
        destination_path=str(destination_root / 'sdxl' / 'base' / 'test_model.safetensors'),
        resolved_url='https://example.com/model.safetensors',
        headers=(('Authorization', 'Bearer token'),),
    )

    calls = {}

    def fake_download_file(**kwargs):
        calls.update(kwargs)
        target = Path(kwargs['model_dir']) / kwargs['file_name']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('ok', encoding='utf-8')
        return str(target)

    monkeypatch.setattr('modules.model_download.transport.download_file', fake_download_file)

    result = Aria2Transport().download(plan)

    assert result.success is True
    assert result.transport == 'aria2'
    assert result.skipped is False
    assert result.destination_path.endswith('test_model.safetensors')
    assert calls['url'] == 'https://example.com/model.safetensors'
    assert calls['prefer_aria2'] is True
    assert calls['headers'] == (('Authorization', 'Bearer token'),)
    assert calls['file_name'] == 'test_model.safetensors'


def test_fallback_transport_disables_aria2(tmp_path, monkeypatch):
    entry = _entry()
    destination_root = tmp_path / 'checkpoints'
    plan = DownloadPlan(
        entry=entry,
        destination_root=str(destination_root),
        destination_path=str(destination_root / 'sdxl' / 'base' / 'test_model.safetensors'),
        resolved_url='https://example.com/model.safetensors',
    )

    calls = {}

    def fake_download_file(**kwargs):
        calls.update(kwargs)
        target = Path(kwargs['model_dir']) / kwargs['file_name']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('ok', encoding='utf-8')
        return str(target)

    monkeypatch.setattr('modules.model_download.transport.download_file', fake_download_file)

    result = FallbackTransport().download(plan)

    assert result.success is True
    assert result.transport == 'fallback'
    assert calls['prefer_aria2'] is False


def test_orchestrator_uses_transport_and_resolver(tmp_path):
    entry = _entry()
    catalog = MagicMock()
    catalog.get.return_value = entry

    class DummyResolver:
        def resolve(self, resolved_entry, policy):
            assert resolved_entry is entry
            destination_root = policy.resolve_root_path(entry)
            return DownloadPlan(
                entry=entry,
                destination_root=destination_root,
                destination_path=os.path.join(destination_root, entry.relative_path),
                resolved_url='https://example.com/model.safetensors',
            )

    class DummyTransport:
        def __init__(self):
            self.plan = None

        def download(self, plan):
            self.plan = plan
            return MagicMock(success=True, destination_path=plan.destination_path, transport='dummy')

    root = tmp_path / 'models'
    root.mkdir()
    policy = ModelDownloadPolicy(root_map={'checkpoints': str(root)})
    transport = DummyTransport()
    orchestrator = ModelDownloadOrchestrator(catalog=catalog, policy=policy, resolver=DummyResolver(), transport=transport)

    result = orchestrator.download('entry.test')

    assert result.success is True
    assert transport.plan.entry is entry


def test_civitai_resolver_appends_token_for_api_download_url(tmp_path, monkeypatch):
    entry = ModelCatalogEntry(
        id='entry.civitai',
        name='model.safetensors',
        root_key='checkpoints',
        relative_path='sdxl/base/model.safetensors',
        architecture='sdxl',
        sub_architecture='base',
        compatibility_family='sdxl',
        token_required=True,
        source=ModelSource(url='https://civitai.com/api/download/models/12345'),
        source_provider='civitai',
    )
    monkeypatch.setenv('CIVITAI_TOKEN', 'secret-token')
    policy = ModelDownloadPolicy(root_map={'checkpoints': str(tmp_path)})

    plan = CivitAIResolver().resolve(entry, policy)

    assert plan.resolved_url == 'https://civitai.com/api/download/models/12345?token=secret-token'
    assert plan.transport == 'civitai_aria2'


def test_github_and_huggingface_resolvers_select_provider_transports(tmp_path):
    policy = ModelDownloadPolicy(root_map={'checkpoints': str(tmp_path)})

    assert GitHubResolver().resolve(_entry(), policy).transport == 'github_aria2'
    assert HuggingFaceResolver().resolve(_entry(), policy).transport == 'hf_get'


def test_runtime_fallback_forwards_headers_to_python_downloader(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: None)

    def fake_load_file_from_url(**kwargs):
        calls.update(kwargs)
        target = Path(kwargs['model_dir']) / kwargs['file_name']
        target.parent.mkdir(parents=True, exist_ok=True)
        header = b'{"value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        target.write_bytes(len(header).to_bytes(8, byteorder='little') + header + b'\x00')
        return str(target)

    monkeypatch.setattr('modules.model_download.runtime.load_file_from_url', fake_load_file_from_url)

    result = download_file(
        url='https://huggingface.co/example/model.safetensors',
        model_dir=str(tmp_path),
        file_name='model.safetensors',
        headers=(('Authorization', 'Bearer hf-token'),),
        prefer_aria2=False,
    )

    assert result.endswith('model.safetensors')
    assert calls['headers'] == (('Authorization', 'Bearer hf-token'),)


def test_runtime_builds_civitai_direct_aria2_command(tmp_path):
    from modules.model_download.runtime import _build_civitai_aria2_command, _ARIA2_USER_AGENT

    command = _build_civitai_aria2_command(
        direct_url='https://b2.civitai.com/file/model.safetensors?Authorization=signed',
        model_dir=str(tmp_path),
        file_name='model.safetensors',
    )

    assert f'--user-agent={_ARIA2_USER_AGENT}' in command
    assert '--check-certificate=false' in command
    assert '-x' in command
    assert '16' in command
    assert '-s' in command
    assert '-c' not in command
    assert command[-1].startswith('https://b2.civitai.com/file/')


def test_runtime_builds_default_aria2_command_for_non_civitai(tmp_path):
    from modules.model_download.runtime import _build_generic_aria2_command

    command = _build_generic_aria2_command(
        url='https://example.com/model.safetensors',
        model_dir=str(tmp_path),
        file_name='model.safetensors',
        headers=(),
    )

    assert '-x' in command
    assert '4' in command
    assert '-s' in command
    assert '-c' in command


def test_runtime_civitai_download_resolves_direct_url_before_aria2(tmp_path, monkeypatch):
    events = []

    monkeypatch.setattr('modules.model_download.runtime.shutil.which', lambda _: 'aria2c')

    def fake_resolve(url, headers=()):
        events.append(('resolve', url, headers))
        return 'https://b2.civitai.com/file/model.safetensors?Authorization=signed'

    def fake_cleanup(path):
        events.append(('cleanup', path))

    def fake_run(command):
        events.append(('run', command))
        target = tmp_path / 'model.safetensors'
        header = b'{"value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        target.write_bytes(len(header).to_bytes(8, byteorder='little') + header + b'\x00')

    monkeypatch.setattr('modules.model_download.runtime._resolve_civitai_direct_url', fake_resolve)
    monkeypatch.setattr('modules.model_download.runtime._cleanup_partial_download', fake_cleanup)
    monkeypatch.setattr('modules.model_download.runtime._run_aria2_command', fake_run)

    result = download_file(
        url='https://civitai.com/api/download/models/12345?token=secret-token',
        model_dir=str(tmp_path),
        file_name='model.safetensors',
        prefer_aria2=True,
    )

    assert result.endswith('model.safetensors')
    assert events[0][0] == 'resolve'
    assert events[1][0] == 'cleanup'
    assert events[2][0] == 'run'
    assert events[2][1][-1].startswith('https://b2.civitai.com/file/')
