import os
import time
import urllib.request
from urllib.parse import urlparse
from typing import Iterable, Optional


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        retries: int = 3,
        retry_delay: float = 2.0,
        headers: Iterable[tuple[str, str]] = (),
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    partial_file = f"{cached_file}.downloading"
    if not os.path.exists(cached_file):
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                if os.path.exists(partial_file):
                    os.remove(partial_file)
                print(f'Downloading: "{url}" to {cached_file} (attempt {attempt}/{retries})\n')
                if headers:
                    _download_url_to_file_with_headers(url, partial_file, headers=headers)
                else:
                    from torch.hub import download_url_to_file
                    download_url_to_file(url, partial_file, progress=progress)
                os.replace(partial_file, cached_file)
                break
            except Exception as exc:
                last_error = exc
                if os.path.exists(partial_file):
                    os.remove(partial_file)
                if attempt >= retries:
                    raise
                print(f"Download failed for {url}: {exc}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        if not os.path.exists(cached_file) and last_error is not None:
            raise last_error
    return cached_file


def _download_url_to_file_with_headers(url: str, destination: str, *, headers: Iterable[tuple[str, str]] = ()) -> None:
    request_headers = {key: value for key, value in headers}
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, 'wb') as target:
        total_size = int(response.headers.get('Content-Length') or 0)
        downloaded = 0
        last_progress_at = time.monotonic()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            downloaded += len(chunk)
            if total_size and time.monotonic() - last_progress_at >= 1:
                print(
                    f'\r{os.path.basename(destination)}: {downloaded / total_size:.1%} '
                    f'({downloaded / 1024 ** 3:.2f}G/{total_size / 1024 ** 3:.2f}G)',
                    end='',
                    flush=True,
                )
                last_progress_at = time.monotonic()
        if total_size:
            print(
                f'\r{os.path.basename(destination)}: 100.0% '
                f'({downloaded / 1024 ** 3:.2f}G/{total_size / 1024 ** 3:.2f}G)'
            )
