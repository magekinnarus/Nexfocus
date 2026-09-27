"""Monotonic revisions and grouped, reversible transaction records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping

from .ids import make_id, validate_id


class RevisionConflict(ValueError):
    """Raised when a mutation was based on an old document revision."""

    def __init__(self, expected: int, current: int) -> None:
        super().__init__(f"expected revision {expected}, current revision is {current}")
        self.expected = expected
        self.current = current


class HistoryValidationError(ValueError):
    """Raised for malformed or non-monotonic history."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class TransactionRecord:
    transaction_id: str
    group_id: str
    actor_id: str
    actor_kind: str
    command_ids: list[str]
    previous_revision: int
    resulting_revision: int
    timestamp: str = field(default_factory=utc_now)
    affected_ids: list[str] = field(default_factory=list)
    before: Any = None
    after: Any = None
    delta: Any = None
    kind: str = "edit"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, previous: "TransactionRecord | None" = None) -> None:
        validate_id(self.transaction_id, field="transaction_id")
        validate_id(self.group_id, field="group_id")
        if not self.actor_id:
            raise HistoryValidationError("actor_id is required")
        if self.actor_kind not in {"human", "agent", "system", "director"}:
            raise HistoryValidationError("invalid actor kind")
        if self.previous_revision < 0 or self.resulting_revision <= self.previous_revision:
            raise HistoryValidationError("transaction revisions must advance")
        if previous is not None and self.previous_revision != previous.resulting_revision:
            raise HistoryValidationError("transaction chain has a revision gap")
        if len(set(self.command_ids)) != len(self.command_ids):
            raise HistoryValidationError("duplicate command id in transaction")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "transactionId": self.transaction_id,
            "groupId": self.group_id,
            "actorId": self.actor_id,
            "actorKind": self.actor_kind,
            "commandIds": list(self.command_ids),
            "previousRevision": self.previous_revision,
            "resultingRevision": self.resulting_revision,
            "timestamp": self.timestamp,
            "affectedIds": list(self.affected_ids),
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransactionRecord":
        required = {
            "transactionId", "groupId", "actorId", "actorKind", "commandIds",
            "previousRevision", "resultingRevision", "timestamp", "affectedIds",
            "before", "after", "delta", "kind", "metadata",
        }
        if set(value) != required:
            raise HistoryValidationError("transaction has unknown or missing fields")
        record = cls(
            transaction_id=str(value["transactionId"]),
            group_id=str(value["groupId"]),
            actor_id=str(value["actorId"]),
            actor_kind=str(value["actorKind"]),
            command_ids=[str(v) for v in value["commandIds"]],
            previous_revision=int(value["previousRevision"]),
            resulting_revision=int(value["resultingRevision"]),
            timestamp=str(value["timestamp"]),
            affected_ids=[str(v) for v in value["affectedIds"]],
            before=value["before"],
            after=value["after"],
            delta=value["delta"],
            kind=str(value["kind"]),
            metadata=dict(value["metadata"]),
        )
        record.validate()
        return record


def validate_history(records: Iterable[TransactionRecord], current_revision: int | None = None) -> list[TransactionRecord]:
    ordered = list(records)
    previous = None
    for record in ordered:
        record.validate(previous=previous)
        previous = record
    if current_revision is not None:
        expected = ordered[-1].resulting_revision if ordered else 0
        if expected != current_revision:
            raise HistoryValidationError(f"history ends at {expected}, document is at {current_revision}")
    return ordered


@dataclass
class RevisionManager:
    """Small in-process revision engine useful to the document core and tests."""

    state: Any
    revision: int = 0
    records: list[TransactionRecord] = field(default_factory=list)
    _snapshots: dict[int, Any] = field(default_factory=dict)
    _undo_stack: list[TransactionRecord] = field(default_factory=list)
    _redo_stack: list[TransactionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._snapshots.setdefault(self.revision, self.state)
        validate_history(self.records, self.revision)
        self._undo_stack = [record for record in self.records if record.kind not in {"undo", "redo"}]

    def commit(
        self,
        expected_revision: int,
        new_state: Any,
        *,
        actor_id: str,
        actor_kind: str,
        command_ids: Iterable[str] = (),
        transaction_id: str | None = None,
        group_id: str | None = None,
        affected_ids: Iterable[str] = (),
        before: Any = None,
        delta: Any = None,
        kind: str = "edit",
        metadata: Mapping[str, Any] | None = None,
    ) -> TransactionRecord:
        if expected_revision != self.revision:
            raise RevisionConflict(expected_revision, self.revision)
        tx = TransactionRecord(
            transaction_id=transaction_id or make_id("txn"),
            group_id=group_id or make_id("grp"),
            actor_id=actor_id,
            actor_kind=actor_kind,
            command_ids=list(command_ids),
            previous_revision=self.revision,
            resulting_revision=self.revision + 1,
            affected_ids=list(affected_ids),
            before=self.state if before is None else before,
            after=new_state,
            delta=delta,
            kind=kind,
            metadata=dict(metadata or {}),
        )
        tx.validate(previous=self.records[-1] if self.records else None)
        self.state = new_state
        self.revision = tx.resulting_revision
        self.records.append(tx)
        self._snapshots[self.revision] = new_state
        if kind == "edit":
            self._undo_stack.append(tx)
            self._redo_stack.clear()
        return tx

    def undo(self, expected_revision: int, *, actor_id: str, actor_kind: str = "human") -> TransactionRecord:
        if expected_revision != self.revision:
            raise RevisionConflict(expected_revision, self.revision)
        if not self._undo_stack:
            raise HistoryValidationError("nothing to undo")
        target = self._undo_stack[-1]
        transaction = self.commit(
            expected_revision,
            target.before,
            actor_id=actor_id,
            actor_kind=actor_kind,
            command_ids=[make_id("undo")],
            kind="undo",
            before=self.state,
            delta={"replayOf": target.transaction_id},
            metadata={"replayOf": target.transaction_id},
        )
        self._undo_stack.pop()
        self._redo_stack.append(target)
        return transaction

    def redo(self, expected_revision: int, *, actor_id: str, actor_kind: str = "human") -> TransactionRecord:
        if expected_revision != self.revision:
            raise RevisionConflict(expected_revision, self.revision)
        if not self._redo_stack:
            raise HistoryValidationError("nothing to redo")
        target = self._redo_stack[-1]
        if target.after is None:
            raise HistoryValidationError("transaction has no replayable after state")
        transaction = self.commit(
            expected_revision,
            target.after,
            actor_id=actor_id,
            actor_kind=actor_kind,
            command_ids=[make_id("redo")],
            kind="redo",
            before=self.state,
            delta={"replayOf": target.transaction_id},
            metadata={"replayOf": target.transaction_id},
        )
        self._redo_stack.pop()
        self._undo_stack.append(target)
        return transaction


HistoryManager = RevisionManager
RevisionLedger = RevisionManager


def grouped_transaction(
    manager: RevisionManager,
    expected_revision: int,
    operations: Iterable[Callable[[Any], Any]],
    *,
    actor_id: str,
    actor_kind: str,
    transaction_id: str | None = None,
    group_id: str | None = None,
    command_ids: Iterable[str] = (),
) -> TransactionRecord:
    """Apply a logical batch to a copy supplied by the caller's operations.

    Operations are evaluated before the manager mutates its state. If one
    raises, no revision or transaction is created.
    """

    candidate = deepcopy(manager.state)
    for operation in operations:
        candidate = operation(candidate)
    return manager.commit(
        expected_revision,
        candidate,
        actor_id=actor_id,
        actor_kind=actor_kind,
        transaction_id=transaction_id,
        group_id=group_id,
        command_ids=command_ids,
    )
