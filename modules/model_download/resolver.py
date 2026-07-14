from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .policy import ModelDownloadPolicy
from .spec import DownloadPlan, ModelCatalogEntry, ModelSource


class DownloadResolver(ABC):
    @abstractmethod
    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        raise NotImplementedError


class DirectResolver(DownloadResolver):
    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        source = _entry_source(entry)
        destination_root = policy.resolve_root_path(entry)
        destination_path = os.path.join(destination_root, entry.relative_path)
        return DownloadPlan(
            entry=entry,
            destination_root=destination_root,
            destination_path=destination_path,
            resolved_url=source.url,
            headers=source.headers,
            transport='generic_aria2',
        )


class GitHubResolver(DirectResolver):
    """Resolve public GitHub Release/raw URLs without provider auth or URL rewriting."""

    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        plan = super().resolve(entry, policy)
        return DownloadPlan(
            entry=plan.entry,
            destination_root=plan.destination_root,
            destination_path=plan.destination_path,
            resolved_url=plan.resolved_url,
            headers=plan.headers,
            transport='github_aria2',
        )


class CivitAIResolver(DownloadResolver):
    def __init__(self, token_env: str = 'CIVITAI_TOKEN'):
        self.token_env = token_env

    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        source = _entry_source(entry)
        token = os.getenv(self.token_env, '')
        resolved_url = source.url
        if entry.token_required and token:
            resolved_url = f'{resolved_url}{"&" if "?" in resolved_url else "?"}token={token}'

        destination_root = policy.resolve_root_path(entry)
        destination_path = os.path.join(destination_root, entry.relative_path)
        return DownloadPlan(
            entry=entry,
            destination_root=destination_root,
            destination_path=destination_path,
            resolved_url=resolved_url,
            headers=source.headers,
            transport='civitai_aria2',
        )


class HuggingFaceResolver(DownloadResolver):
    def __init__(self, token_env: str = 'HUGGINGFACE_TOKEN'):
        self.token_env = token_env

    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        source = _entry_source(entry)
        headers = list(source.headers)
        token = os.getenv(self.token_env, '')
        if entry.token_required and token:
            headers.append(('Authorization', f'Bearer {token}'))

        destination_root = policy.resolve_root_path(entry)
        destination_path = os.path.join(destination_root, entry.relative_path)
        return DownloadPlan(
            entry=entry,
            destination_root=destination_root,
            destination_path=destination_path,
            resolved_url=source.url,
            headers=tuple(headers),
            transport='hf_get',
        )


def _entry_source(entry: ModelCatalogEntry) -> ModelSource:
    if entry.source is None:
        raise ValueError(f'Catalog entry {entry.id} does not define a source')
    return entry.source
