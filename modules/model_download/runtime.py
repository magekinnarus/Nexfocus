from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from modules.model_loader import load_file_from_url

try:
    import requests
except ImportError:  # pragma: no cover - fallback path only matters if requests is missing.
    requests = None

_ARIA2_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
_ARIA2_GENERIC_CONNECTIONS = '4'
_ARIA2_CIVITAI_CONNECTIONS = '16'
_ARIA2_GITHUB_CONNECTIONS = '16'
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


class _CivitAIAuthenticationError(RuntimeError):
    pass


def download_file(
    url: str,
    *,
    model_dir: str,
    file_name: str | None = None,
    progress: bool = True,
    headers: Iterable[tuple[str, str]] = (),
    prefer_aria2: bool = True,
) -> str:
    if not file_name:
        file_name = os.path.basename(urlparse(url).path)
    else:
        file_name = file_name.replace('\\', '/')
        if '/' in file_name:
            sub_dir, file_name = file_name.rsplit('/', 1)
            model_dir = os.path.join(model_dir, sub_dir.replace('/', os.sep))

    os.makedirs(model_dir, exist_ok=True)

    destination = os.path.abspath(os.path.join(model_dir, file_name))
    partial_download = os.path.exists(f'{destination}.aria2')
    if os.path.exists(destination) and not partial_download:
        if validate_downloaded_file(destination):
            return destination
        print(f"Discarding invalid cached model payload: {destination}")
        _cleanup_partial_download(destination)

    if _is_huggingface_download_url(url):
        # HF downloads use one streaming GET. Aria2's redirect/Xet handling is
        # intentionally bypassed, and stale Aria2 state is not resumed.
        if partial_download:
            _cleanup_partial_download(destination)
        downloaded = load_file_from_url(
            url=_with_download_query(url),
            model_dir=model_dir,
            file_name=file_name,
            progress=progress,
            headers=headers,
        )
        return _require_valid_download(downloaded, url=url)

    if prefer_aria2 and shutil.which('aria2c'):
        try:
            downloaded = _download_with_aria2(
                url=url,
                model_dir=model_dir,
                file_name=file_name,
                headers=headers,
            )
            return _require_valid_download(downloaded, url=url)
        except Exception as exc:
            if isinstance(exc, _CivitAIAuthenticationError):
                raise
            if _is_huggingface_download_url(url):
                print(
                    f"Aria2 download failed for Hugging Face asset {url}: {exc}. "
                    "Keeping any Aria2 partial files and not falling back to the slow Python downloader."
                )
                raise
            print(f"Aria2 download failed for {url}: {exc}. Falling back to the Python downloader.")
            _cleanup_partial_download(destination)

    if partial_download:
        _cleanup_partial_download(destination)

    downloaded = load_file_from_url(
        url=url,
        model_dir=model_dir,
        file_name=file_name,
        progress=progress,
        headers=headers,
    )
    return _require_valid_download(downloaded, url=url)


def validate_downloaded_file(path: str) -> bool:
    """Cheaply reject obvious non-model payloads saved under model filenames."""
    if not str(path).lower().endswith('.safetensors'):
        return True

    try:
        file_size = os.path.getsize(path)
        if file_size < 10:
            return False

        with open(path, 'rb') as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                return False
            header_size = int.from_bytes(header_size_raw, byteorder='little', signed=False)
            if not 2 <= header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
                return False
            if header_size > file_size - 8:
                return False
            header = handle.read(header_size)

        metadata = json.loads(header.decode('utf-8'))
        if not isinstance(metadata, dict):
            return False

        payload_size = file_size - 8 - header_size
        tensor_descriptors = [
            descriptor
            for key, descriptor in metadata.items()
            if key != '__metadata__'
        ]
        if not tensor_descriptors:
            return False

        for descriptor in tensor_descriptors:
            if not isinstance(descriptor, dict):
                return False
            if not isinstance(descriptor.get('dtype'), str):
                return False
            if not isinstance(descriptor.get('shape'), list):
                return False
            offsets = descriptor.get('data_offsets')
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(value, int) for value in offsets)
                and 0 <= offsets[0] <= offsets[1] <= payload_size
            ):
                return False
        return True
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False


def _require_valid_download(path: str, *, url: str) -> str:
    if validate_downloaded_file(path):
        return path

    _cleanup_partial_download(path)
    raise RuntimeError(
        f"Downloaded payload from {url} is not a valid safetensors file. "
        "The response may be an HTML/XML error page or an unresolved pointer."
    )


def _is_civitai_api_download_url(url: str) -> bool:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    path = (parsed.path or '').lower()
    return host.endswith('civitai.com') and '/api/download/models/' in path


def _resolve_civitai_direct_url(url: str, headers: Iterable[tuple[str, str]] = ()) -> str:
    url = _with_civitai_token(url)
    request_headers = {key: value for key, value in headers}
    if 'User-Agent' not in request_headers:
        request_headers['User-Agent'] = _ARIA2_USER_AGENT

    def _do_resolve(target_url: str) -> str:
        try:
            if requests is not None:
                try:
                    response = requests.head(target_url, headers=request_headers, allow_redirects=True, timeout=10)
                    response.raise_for_status()
                    return _validate_civitai_direct_response(
                        response.url,
                        content_type=response.headers.get('Content-Type'),
                    )
                except Exception as head_exc:
                    status_code = getattr(getattr(head_exc, 'response', None), 'status_code', None)
                    if status_code in (403, 405) or "403" in str(head_exc) or "405" in str(head_exc):
                        with requests.get(target_url, headers=request_headers, allow_redirects=True, timeout=10, stream=True) as response:
                            response.raise_for_status()
                            return _validate_civitai_direct_response(
                                response.url,
                                content_type=response.headers.get('Content-Type'),
                            )
                    raise head_exc

            import urllib.request
            request = urllib.request.Request(target_url, headers=request_headers, method='HEAD')
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return _validate_civitai_direct_response(
                        response.geturl(),
                        content_type=response.headers.get('Content-Type'),
                    )
            except Exception:
                request.method = 'GET'
                with urllib.request.urlopen(request, timeout=10) as response:
                    return _validate_civitai_direct_response(
                        response.geturl(),
                        content_type=response.headers.get('Content-Type'),
                    )
        except Exception as exc:
            if isinstance(exc, _CivitAIAuthenticationError):
                raise
            parsed = urlparse(target_url)
            if parsed.netloc.lower() == 'civitai.com':
                red_url = target_url.replace('civitai.com', 'civitai.red', 1)
                print(f"CivitAI .com resolution failed for {target_url}: {exc}. Retrying via .red ...")
                return _do_resolve(red_url)
            raise exc

    return _do_resolve(url)


def _with_civitai_token(url: str) -> str:
    parsed = urlparse(str(url or '').strip())
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key.lower() == 'token' and value for key, value in query_items):
        return url

    token = os.environ.get('CIVITAI_TOKEN', '').strip()
    if not token:
        return url

    query_items.append(('token', token))
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def _validate_civitai_direct_response(url: str, *, content_type: str | None = None) -> str:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    path = (parsed.path or '').lower()
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    normalized_content_type = str(content_type or '').lower()

    is_auth_redirect = (
        host == 'auth.civitai.com'
        or path.startswith('/login')
        or query.get('reason') == 'download-auth'
    )
    if is_auth_redirect or 'text/html' in normalized_content_type:
        raise _CivitAIAuthenticationError(
            'CivitAI returned an authentication/login page instead of a model file. '
            'Set CIVITAI_TOKEN to a valid CivitAI API token and retry.'
        )
    return url


def _is_huggingface_download_url(url: str) -> bool:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    mirror_host = urlparse(os.environ.get("HF_MIRROR", "")).netloc.lower()
    return host.endswith('huggingface.co') or (mirror_host and host == mirror_host)


def _is_github_download_url(url: str) -> bool:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    return (
        host == 'github.com'
        or host.endswith('.github.com')
        or host == 'githubusercontent.com'
        or host.endswith('.githubusercontent.com')
    )


def _with_download_query(url: str) -> str:
    parsed = urlparse(str(url or '').strip())
    if not parsed.netloc:
        return url

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key.lower() == 'download' for key, _value in query_items):
        query_items = [
            (key, 'true') if key.lower() == 'download' else (key, value)
            for key, value in query_items
        ]
    else:
        query_items.append(('download', 'true'))
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def _build_generic_aria2_command(
    *,
    url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
    connections: str = _ARIA2_GENERIC_CONNECTIONS,
) -> list[str]:
    command = [
        'aria2c',
        '--console-log-level=warn',
        '--max-tries=20',
        '--retry-wait=5',
        '--timeout=60',
        '--connect-timeout=30',
        '--file-allocation=none',
        '-c',
        '-x', connections,
        '-s', connections,
        '-k', '1M',
        '--dir', model_dir,
        '--out', file_name,
    ]

    for key, value in headers:
        command.extend(['--header', f'{key}: {value}'])

    command.append(url)
    return command


def _build_github_aria2_command(
    *,
    url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
) -> list[str]:
    return _build_generic_aria2_command(
        url=url,
        model_dir=model_dir,
        file_name=file_name,
        headers=headers,
        connections=_ARIA2_GITHUB_CONNECTIONS,
    )


def _build_civitai_aria2_command(*, direct_url: str, model_dir: str, file_name: str) -> list[str]:
    return [
        'aria2c',
        '--console-log-level=warn',
        '--max-tries=20',
        '--retry-wait=5',
        '--timeout=60',
        '--connect-timeout=30',
        '--file-allocation=none',
        f'--user-agent={_ARIA2_USER_AGENT}',
        '--check-certificate=false',
        '-x', _ARIA2_CIVITAI_CONNECTIONS,
        '-s', _ARIA2_CIVITAI_CONNECTIONS,
        '-k', '1M',
        '--dir', model_dir,
        '--out', file_name,
        direct_url,
    ]


def _run_aria2_command(command: list[str]) -> None:
    subprocess.check_call(command)


def _download_with_aria2(
    *,
    url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
) -> str:
    destination = os.path.abspath(os.path.join(model_dir, file_name))
    headers = tuple(headers)

    if _is_huggingface_download_url(url):
        raise RuntimeError('Hugging Face assets must use the single-stream downloader, not Aria2.')

    if _is_civitai_api_download_url(url):
        direct_url = _resolve_civitai_direct_url(url, headers=headers)
        _cleanup_partial_download(destination)
        command = _build_civitai_aria2_command(
            direct_url=direct_url,
            model_dir=model_dir,
            file_name=file_name,
        )
    elif _is_github_download_url(url):
        command = _build_github_aria2_command(
            url=url,
            model_dir=model_dir,
            file_name=file_name,
            headers=headers,
        )
    else:
        command = _build_generic_aria2_command(
            url=url,
            model_dir=model_dir,
            file_name=file_name,
            headers=headers,
        )

    _run_aria2_command(command)
    return destination


def _cleanup_partial_download(destination: str) -> None:
    for candidate in (destination, f'{destination}.aria2'):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass
