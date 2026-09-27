"""Public, renderer-independent creative-document foundation.

The package owns durable records, identity, transforms, history, assets, and
native project persistence. It intentionally has no Gradio, browser, model,
inference, or transport dependency.
"""

from .asset_store import AssetHashMismatch, AssetStore, AssetStoreError, MissingAsset
from .history import (
    HistoryManager,
    HistoryValidationError,
    RevisionConflict,
    RevisionLedger,
    RevisionManager,
    TransactionRecord,
    grouped_transaction,
)
from .ids import canonical_json, content_digest, make_id, new_id, sha256_bytes, sha256_file, validate_id
from .migrations import MigrationError, MigrationRegistry, UnsupportedSchemaVersion, migrate_manifest
from .project_store import CorruptProject, FaultInjector, InjectedInterruption, ProjectStore, ProjectStoreError, SaveResult
from .schema import (
    BBOperation,
    BBOperationRecord,
    AssetRecord,
    Candidate,
    CandidateRecord,
    CollaborationState,
    CorrectivePatch,
    DepthComposite,
    Document,
    DocumentRecord,
    ExtractionDerivative,
    ExtractionRecord,
    ExternalRoundTrip,
    GuideLifecycle,
    GuideRecord,
    HandoffNote,
    InteractionGroup,
    Layer,
    LayerRecord,
    Mask,
    MaskRecord,
    Object,
    ObjectRecord,
    OperationRecord,
    PrivateProxy,
    PrivateProxyRecord,
    SchemaValidationError,
    Selection,
    SelectionRecord,
    SelectionState,
    VariantSet,
    VisualProfile,
)
from .transforms import (
    AffineTransform,
    BBox,
    CoordinateTransform,
    Space,
    Transform,
    TransformError,
    TransformRecord,
    map_pixel_center,
    pixel_center,
    unmap_pixel_center,
)
from .tree import TreeValidationError, ordered_layer_ids, validate_layer_tree, validate_tree

__all__ = [
    "AffineTransform", "AssetHashMismatch", "AssetRecord", "AssetStore", "AssetStoreError", "BBOperation", "BBOperationRecord", "BBox",
    "Candidate", "CandidateRecord", "CollaborationState", "CoordinateTransform", "CorrectivePatch", "CorruptProject", "DepthComposite",
    "Document", "DocumentRecord", "ExtractionDerivative", "ExtractionRecord", "ExternalRoundTrip", "FaultInjector", "GuideLifecycle", "GuideRecord",
    "grouped_transaction", "HandoffNote", "HistoryManager", "HistoryValidationError", "InjectedInterruption", "InteractionGroup", "Layer", "LayerRecord",
    "Mask", "MaskRecord", "MigrationError", "MigrationRegistry", "MissingAsset", "Object", "ObjectRecord", "OperationRecord", "PrivateProxy", "PrivateProxyRecord",
    "ProjectStore", "ProjectStoreError", "RevisionConflict", "RevisionLedger", "RevisionManager", "SaveResult", "SchemaValidationError", "Selection", "SelectionRecord",
    "SelectionState", "sha256_bytes", "sha256_file", "Space", "TransactionRecord", "Transform", "TransformError", "TransformRecord", "TreeValidationError",
    "UnsupportedSchemaVersion", "VariantSet", "VisualProfile", "canonical_json", "content_digest", "make_id", "map_pixel_center", "migrate_manifest", "new_id",
    "ordered_layer_ids", "pixel_center", "unmap_pixel_center", "validate_id", "validate_layer_tree", "validate_tree",
]
