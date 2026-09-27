"""Monotonic revisions and grouped, reversible transaction records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping

from .ids import canonical_json, make_id, validate_id


class RevisionConflict(ValueError):
    """Raised when a mutation was based on an old document revision."""

    def __init__(self, expected: int, current: int) -> None:
        super().__init__(f"expected revision {expected}, current revision is {current}")
        self.expected = expected
        self.current = current


class HistoryValidationError(ValueError):
    """Raised for malformed or non-monotonic history."""


class SelectionRevisionConflict(HistoryValidationError):
    """A selection mutation was based on a stale selection revision."""

    def __init__(self, expected: int, current: int) -> None:
        super().__init__(f"expected selection revision {expected}, current selection revision is {current}")
        self.expected = expected
        self.current = current


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
        try:
            validate_id(self.transaction_id, field="transaction_id")
            validate_id(self.group_id, field="group_id")
            validate_id(self.actor_id, field="actor_id")
        except (TypeError, ValueError) as exc:
            raise HistoryValidationError(str(exc)) from exc
        if self.actor_kind not in {"human", "agent", "system", "director"}:
            raise HistoryValidationError("invalid actor kind")
        if (type(self.previous_revision) is not int or type(self.resulting_revision) is not int
                or self.previous_revision < 0 or self.resulting_revision != self.previous_revision + 1):
            raise HistoryValidationError("each transaction must advance exactly one revision")
        if previous is not None and self.previous_revision != previous.resulting_revision:
            raise HistoryValidationError("transaction chain has a revision gap")
        if not isinstance(self.command_ids, list) or not self.command_ids:
            raise HistoryValidationError("transaction requires at least one command ID")
        if not isinstance(self.affected_ids, list):
            raise HistoryValidationError("affected IDs must be an array")
        for value in [*self.command_ids, *self.affected_ids]:
            try:
                validate_id(value, field="command/affected ID")
            except (TypeError, ValueError) as exc:
                raise HistoryValidationError(str(exc)) from exc
        if len(set(self.command_ids)) != len(self.command_ids):
            raise HistoryValidationError("duplicate command id in transaction")
        if len(set(self.affected_ids)) != len(self.affected_ids):
            raise HistoryValidationError("duplicate affected ID in transaction")
        if self.kind not in {"edit", "undo", "redo", "selection-create", "selection-refine",
                             "selection-rebase", "selection-duplicate", "source-invalidation"}:
            raise HistoryValidationError("invalid transaction kind")
        if not isinstance(self.timestamp, str):
            raise HistoryValidationError("transaction timestamp must be an ISO-8601 string")
        try:
            parsed_timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HistoryValidationError("transaction timestamp must be valid ISO-8601") from exc
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
            raise HistoryValidationError("transaction timestamp must include a timezone")
        if self.before is None and self.after is None and not self.delta:
            raise HistoryValidationError("transaction lacks reversible before/after or delta material")
        try:
            canonical_json({"before": self.before, "after": self.after, "delta": self.delta, "metadata": self.metadata})
        except (TypeError, ValueError) as exc:
            raise HistoryValidationError(f"transaction material must be finite JSON data: {exc}") from exc
        if not isinstance(self.metadata, dict):
            raise HistoryValidationError("transaction metadata must be an object")

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
        if not isinstance(value, Mapping) or set(value) != required:
            raise HistoryValidationError("transaction has unknown or missing fields")
        string_fields = ("transactionId", "groupId", "actorId", "actorKind", "timestamp", "kind")
        if any(not isinstance(value[field], str) for field in string_fields):
            raise HistoryValidationError("transaction identity, actor, timestamp, and kind fields must be strings")
        if any(type(value[field]) is not int for field in ("previousRevision", "resultingRevision")):
            raise HistoryValidationError("transaction revisions must be integers")
        if (not isinstance(value["commandIds"], list) or not all(isinstance(v, str) for v in value["commandIds"])
                or not isinstance(value["affectedIds"], list) or not all(isinstance(v, str) for v in value["affectedIds"])):
            raise HistoryValidationError("commandIds and affectedIds must be arrays of strings")
        if not isinstance(value["metadata"], dict):
            raise HistoryValidationError("transaction metadata must be an object")
        record = cls(
            transaction_id=value["transactionId"],
            group_id=value["groupId"],
            actor_id=value["actorId"],
            actor_kind=value["actorKind"],
            command_ids=list(value["commandIds"]),
            previous_revision=value["previousRevision"],
            resulting_revision=value["resultingRevision"],
            timestamp=value["timestamp"],
            affected_ids=list(value["affectedIds"]),
            before=value["before"],
            after=value["after"],
            delta=value["delta"],
            kind=value["kind"],
            metadata=dict(value["metadata"]),
        )
        record.validate()
        return record


def validate_history(records: Iterable[TransactionRecord], current_revision: int | None = None) -> list[TransactionRecord]:
    ordered = list(records)
    previous = None
    transaction_ids: set[str] = set()
    group_ids: set[str] = set()
    for record in ordered:
        record.validate(previous=previous)
        if record.transaction_id in transaction_ids:
            raise HistoryValidationError("duplicate transaction ID in history")
        if record.group_id in group_ids:
            raise HistoryValidationError("duplicate group ID in history")
        transaction_ids.add(record.transaction_id)
        group_ids.add(record.group_id)
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
