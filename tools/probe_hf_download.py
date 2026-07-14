from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SELECTOR = 'innovision_v10.safetensors'
DEFAULT_CATALOG = REPO_ROOT / 'configs' / 'model_catalogs' / 'huggingface_main_catalog.json'
DEFAULT_TARGET_ROOT = REPO_ROOT / '.agent' / 'temp' / 'hf_download_probe'


def _load_runtime_download_file():
    runtime_path = REPO_ROOT / 'modules' / 'model_download' / 'runtime.py'
    spec = importlib.util.spec_from_file_location('nex_probe_model_download_runtime', runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load runtime module from {runtime_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.download_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Probe the project Hugging Face downloader without launching the UI.',
    )
    parser.add_argument(
        '--selector',
        default=DEFAULT_SELECTOR,
        help='Catalog id, alias, display name, or file name to download.',
    )
    parser.add_argument(
        '--catalog',
        default=str(DEFAULT_CATALOG),
        help='Model catalog JSON to resolve --selector from.',
    )
    parser.add_argument(
        '--target-root',
        default=str(DEFAULT_TARGET_ROOT),
        help='Local root for probe downloads. Defaults under .agent/temp.',
    )
    parser.add_argument('--url', help='Direct URL. When set, catalog lookup is skipped.')
    parser.add_argument('--file-name', help='Output filename for --url or catalog override.')
    parser.add_argument('--target-dir', help='Exact directory to download into.')
    parser.add_argument(
        '--idle-timeout',
        type=float,
        default=0.0,
        help='Deprecated compatibility option; the current runtime probe does not use Hugging Face Hub.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Remove the existing destination and known partial files before downloading.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Resolve and print the download target without downloading.',
    )
    parser.add_argument(
        '--sha256',
        action='store_true',
        help='Compute sha256 after download. This can take time for multi-GB files.',
    )
    parser.add_argument(
        '--skip-safetensors-check',
        action='store_true',
        help='Skip the safetensors header validation for .safetensors files.',
    )
    return parser


def _iter_entry_dicts(node):
    if isinstance(node, list):
        for item in node:
            yield from _iter_entry_dicts(item)
    elif isinstance(node, dict):
        if 'id' in node and 'name' in node and 'root_key' in node:
            yield node
        else:
            for value in node.values():
                yield from _iter_entry_dicts(value)


def _select_entry(catalog_path: Path, selector: str) -> dict:
    entries = list(_iter_entry_dicts(json.loads(catalog_path.read_text(encoding='utf-8-sig'))))

    normalized = selector.casefold()
    matches = [
        candidate
        for candidate in entries
        if normalized
        in {
            str(candidate.get('name', '')).casefold(),
            str(candidate.get('display_name', '')).casefold(),
            str(candidate.get('id', '')).casefold(),
            str(candidate.get('alias', '')).casefold(),
        }
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        choices = ', '.join(entry['name'] for entry in matches[:10])
        raise SystemExit(f'Ambiguous selector {selector!r}. Matches include: {choices}')
    raise SystemExit(f'No catalog entry found for selector {selector!r} in {catalog_path}')


def _resolve_download(args: argparse.Namespace) -> tuple[str, Path, str, tuple[tuple[str, str], ...]]:
    if args.url:
        file_name = args.file_name or Path(args.url.split('?', 1)[0]).name
        target_dir = Path(args.target_dir) if args.target_dir else Path(args.target_root)
        return args.url, target_dir, file_name, ()

    entry = _select_entry(Path(args.catalog), args.selector)
    source = entry.get('source') or {}
    if not source.get('url'):
        raise SystemExit(f"Catalog entry {entry.get('id')} does not define a source URL.")

    file_name = args.file_name or entry['name']
    target_dir = Path(args.target_dir) if args.target_dir else Path(args.target_root) / entry['root_key']
    headers = tuple(tuple(header) for header in source.get('headers', []))
    return source['url'], target_dir, file_name, headers


def _remove_existing(destination: Path) -> None:
    for candidate in (
        destination,
        Path(f'{destination}.aria2'),
        Path(f'{destination}.downloading'),
    ):
        if candidate.exists():
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_safetensors(path: Path) -> None:
    if path.suffix.lower() != '.safetensors':
        return
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework='pt', device='cpu') as handle:
            tensor_count = len(handle.keys())
    except Exception as exc:
        raise SystemExit(f'Safetensors header validation failed: {exc}') from exc
    print(f'Safetensors header OK: {tensor_count} tensors')


def _is_huggingface_url(url: str) -> bool:
    host = urlparse(str(url or '').strip()).netloc.lower()
    mirror_host = urlparse(os.environ.get('HF_MIRROR', '')).netloc.lower()
    return host.endswith('huggingface.co') or (mirror_host and host == mirror_host)


def _report_hf_transport(url: str) -> None:
    if not _is_huggingface_url(url):
        return
    print('HF transport: single-thread Python GET; Aria2 is bypassed.')
    print('HF request: download=true; retries use the temporary .downloading file.')
    print('HF Hub/Xet: not used by the project downloader.')


def main() -> int:
    args = _build_parser().parse_args()
    url, target_dir, file_name, headers = _resolve_download(args)
    target_dir = target_dir.resolve()
    destination = target_dir / file_name

    print(f'Python: {sys.executable}')
    print(f'URL: {url}')
    print(f'Destination: {destination}')
    _report_hf_transport(url)

    if args.dry_run:
        return 0

    download_file = _load_runtime_download_file()

    if args.force:
        _remove_existing(destination)

    started = time.perf_counter()
    try:
        result = Path(
            download_file(
                url=url,
                model_dir=str(target_dir),
                file_name=file_name,
                progress=True,
                headers=headers,
                prefer_aria2=True,
            )
        )
    except KeyboardInterrupt:
        print('Download interrupted by user.')
        if _is_huggingface_url(url) and any(
            candidate.exists()
            for candidate in (destination, Path(f'{destination}.aria2'), Path(f'{destination}.downloading'))
        ):
            print('HF temporary download state is not resumable; rerun the single-stream download.')
        return 130
    except Exception as exc:
        print(f'Download failed: {exc}')
        if _is_huggingface_url(url) and any(
            candidate.exists()
            for candidate in (destination, Path(f'{destination}.aria2'), Path(f'{destination}.downloading'))
        ):
            print('HF temporary download state is not resumable; rerun the single-stream download.')
        return 1
    elapsed = time.perf_counter() - started

    size_gib = result.stat().st_size / 1024**3
    print(f'Download complete: {result}')
    print(f'Size: {size_gib:.2f} GiB')
    print(f'Elapsed: {elapsed:.2f}s')

    if not args.skip_safetensors_check:
        _validate_safetensors(result)
    if args.sha256:
        print(f'SHA256: {_sha256(result)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
