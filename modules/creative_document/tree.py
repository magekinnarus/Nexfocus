"""Validation for the ordered, single-parent layer/group tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class TreeValidationError(ValueError):
    """Raised when layer/group ownership or ordering is invalid."""


def _value(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(name, default)
    return getattr(node, name, default)


def validate_layer_tree(layers: Mapping[str, Any] | Sequence[Any], root_ids: Sequence[str] | None = None) -> list[str]:
    """Validate parentage, ordered membership, and cycles.

    ``layers`` may be the document's ID mapping or a sequence of records. The
    document root is an implicit ordered root, so more than one top-level
    layer is valid while every node still belongs to exactly one tree.
    """

    if isinstance(layers, Mapping):
        records = dict(layers)
    else:
        records = {}
        for node in layers:
            node_id = _value(node, "layer_id", _value(node, "id"))
            if node_id in records:
                raise TreeValidationError(f"duplicate layer id: {node_id}")
            records[node_id] = node

    if root_ids is None:
        root_ids = [node_id for node_id, node in records.items() if _value(node, "parent_id") is None]
    roots = list(root_ids)
    if len(set(roots)) != len(roots):
        raise TreeValidationError("duplicate root membership")
    if any(node_id not in records for node_id in roots):
        raise TreeValidationError("root references an unknown layer")

    child_membership: dict[str, str] = {}
    for parent_id, parent in records.items():
        child_ids = list(_value(parent, "child_ids", []) or [])
        if len(set(child_ids)) != len(child_ids):
            raise TreeValidationError(f"duplicate child membership under {parent_id}")
        for child_id in child_ids:
            if child_id not in records:
                raise TreeValidationError(f"{parent_id} references unknown child {child_id}")
            if child_id in child_membership and child_membership[child_id] != parent_id:
                raise TreeValidationError(f"layer {child_id} has multiple parents")
            child_membership[child_id] = parent_id
            declared_parent = _value(records[child_id], "parent_id")
            if declared_parent != parent_id:
                raise TreeValidationError(f"child {child_id} parent mismatch")

    for node_id, node in records.items():
        parent_id = _value(node, "parent_id")
        if parent_id is None:
            if node_id not in roots:
                raise TreeValidationError(f"top-level layer {node_id} missing from root order")
        else:
            if parent_id not in records:
                raise TreeValidationError(f"layer {node_id} references unknown parent {parent_id}")
            if child_membership.get(node_id) != parent_id:
                raise TreeValidationError(f"layer {node_id} is not in its parent's ordered children")

    state: dict[str, int] = {}

    def visit(node_id: str) -> None:
        current = state.get(node_id, 0)
        if current == 1:
            raise TreeValidationError(f"layer tree cycle at {node_id}")
        if current == 2:
            return
        state[node_id] = 1
        for child_id in _value(records[node_id], "child_ids", []) or []:
            visit(child_id)
        state[node_id] = 2

    for root_id in roots:
        visit(root_id)
    if len(state) != len(records):
        missing = sorted(set(records) - set(state))
        raise TreeValidationError(f"layer tree contains unreachable nodes: {missing}")
    return roots


def ordered_layer_ids(layers: Mapping[str, Any], root_ids: Sequence[str]) -> list[str]:
    validate_layer_tree(layers, root_ids)
    result: list[str] = []

    def walk(node_id: str) -> None:
        result.append(node_id)
        for child_id in _value(layers[node_id], "child_ids", []) or []:
            walk(child_id)

    for root_id in root_ids:
        walk(root_id)
    return result


validate_tree = validate_layer_tree
