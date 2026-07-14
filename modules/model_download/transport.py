from __future__ import annotations

import os
from abc import ABC, abstractmethod

from modules.model_download.runtime import download_file

from .spec import DownloadPlan, DownloadResult, ModelCatalogEntry


class DownloadTransport(ABC):
    @abstractmethod
    def download(self, plan: DownloadPlan) -> DownloadResult:
        raise NotImplementedError


class Aria2Transport(DownloadTransport):
    def download(self, plan: DownloadPlan) -> DownloadResult:
        return _download_plan(plan, prefer_aria2=True, transport_name=plan.transport)


class FallbackTransport(DownloadTransport):
    def download(self, plan: DownloadPlan) -> DownloadResult:
        return _download_plan(plan, prefer_aria2=False, transport_name='fallback')


def _download_plan(plan: DownloadPlan, *, prefer_aria2: bool, transport_name: str) -> DownloadResult:
    destination_path = os.path.abspath(plan.destination_path)
    if os.path.exists(destination_path):
        return DownloadResult(
            success=True,
            destination_path=destination_path,
            transport=transport_name,
            message='Already downloaded',
            skipped=True,
        )

    destination_root = os.path.dirname(destination_path) or plan.destination_root
    os.makedirs(destination_root, exist_ok=True)
    file_name = os.path.basename(destination_path)

    try:
        result_path = download_file(
            url=plan.resolved_url,
            model_dir=destination_root,
            file_name=file_name,
            progress=True,
            headers=plan.headers,
            prefer_aria2=prefer_aria2,
        )
        return DownloadResult(
            success=True,
            destination_path=os.path.abspath(result_path),
            transport=transport_name,
            message='Download completed',
        )
    except Exception as exc:
        return DownloadResult(
            success=False,
            destination_path=destination_path,
            transport=transport_name,
            message=str(exc),
        )
