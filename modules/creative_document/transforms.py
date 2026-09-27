"""Renderer-independent coordinate spaces and affine registration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Iterable, Mapping, Sequence


class TransformError(ValueError):
    """Raised for malformed or non-invertible transforms."""


class Space(StrEnum):
    DOCUMENT = "document"
    SOURCE = "document"
    SOURCE_CROP = "source-crop"
    NATIVE_BB = "native-bb"
    MODEL = "model"
    PREVIEW = "preview"


def _finite(value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise TransformError("transform values must be finite")
    return value


@dataclass(frozen=True)
class BBox:
    """A half-open integer pixel rectangle in y1, y2, x1, x2 order."""

    y1: int
    y2: int
    x1: int
    x2: int

    def validate(self, width: int | None = None, height: int | None = None) -> "BBox":
        values = (self.y1, self.y2, self.x1, self.x2)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise TransformError("bbox coordinates must be integers")
        if self.y1 < 0 or self.x1 < 0 or self.y2 <= self.y1 or self.x2 <= self.x1:
            raise TransformError("bbox must be a non-empty half-open rectangle")
        if width is not None and self.x2 > width:
            raise TransformError("bbox exceeds width")
        if height is not None and self.y2 > height:
            raise TransformError("bbox exceeds height")
        return self

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return {"y1": self.y1, "y2": self.y2, "x1": self.x1, "x2": self.x2}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BBox":
        if set(value) != {"y1", "y2", "x1", "x2"}:
            raise TransformError("bbox must contain exactly y1, y2, x1, x2")
        return cls(int(value["y1"]), int(value["y2"]), int(value["x1"]), int(value["x2"])).validate()


@dataclass(frozen=True)
class AffineTransform:
    """A row-major 3x3 homogeneous affine transform."""

    matrix: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.matrix) != 9:
            raise TransformError("an affine matrix must contain nine values")
        values = tuple(_finite(v) for v in self.matrix)
        if abs(values[6]) > 1e-12 or abs(values[7]) > 1e-12 or abs(values[8] - 1.0) > 1e-12:
            raise TransformError("matrix is not affine homogeneous form")
        object.__setattr__(self, "matrix", values)
        if abs(self.determinant()) < 1e-12:
            raise TransformError("singular affine transform")

    @classmethod
    def identity(cls) -> "AffineTransform":
        return cls()

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "AffineTransform":
        return cls(tuple(values))

    @classmethod
    def translation(cls, dx: float, dy: float) -> "AffineTransform":
        return cls((1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0))

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> "AffineTransform":
        sy = sx if sy is None else sy
        return cls((sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0))

    def determinant(self) -> float:
        a, b, _ = self.matrix[:3]
        d, e, _ = self.matrix[3:6]
        return a * e - b * d

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f, *_ = self.matrix
        return (a * x + b * y + c, d * x + e * y + f)

    def inverse(self) -> "AffineTransform":
        a, b, c, d, e, f, *_ = self.matrix
        det = a * e - b * d
        if abs(det) < 1e-12:
            raise TransformError("singular affine transform has no inverse")
        return AffineTransform((
            e / det,
            -b / det,
            (b * f - e * c) / det,
            -d / det,
            a / det,
            (d * c - a * f) / det,
            0.0,
            0.0,
            1.0,
        ))

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Return self after other (``self(other(point))``)."""

        left = self.matrix
        right = other.matrix
        values = tuple(
            sum(left[row * 3 + k] * right[k * 3 + col] for k in range(3))
            for row in range(3)
            for col in range(3)
        )
        return AffineTransform(values)

    def map_bbox(self, bbox: BBox) -> tuple[tuple[float, float], ...]:
        bbox.validate()
        return tuple(
            self.apply_point(x, y)
            for x, y in (
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y1),
                (bbox.x2, bbox.y2),
                (bbox.x1, bbox.y2),
            )
        )

    def to_dict(self) -> list[float]:
        return list(self.matrix)

    @classmethod
    def from_dict(cls, value: Sequence[float]) -> "AffineTransform":
        return cls.from_values(value)


@dataclass(frozen=True)
class CoordinateTransform:
    """A persisted transform with named spaces and dimensional provenance."""

    from_space: str
    to_space: str
    source_dimensions: tuple[int, int]
    target_dimensions: tuple[int, int]
    forward: AffineTransform
    inverse: AffineTransform
    operation_revision: int | str | None = None

    def __post_init__(self) -> None:
        if not self.from_space or not self.to_space:
            raise TransformError("coordinate spaces are required")
        for dimensions in (self.source_dimensions, self.target_dimensions):
            if len(dimensions) != 2 or any(int(v) <= 0 for v in dimensions):
                raise TransformError("transform dimensions must be positive width/height")
        if self.forward.compose(self.inverse).matrix != AffineTransform.identity().matrix:
            # Floating-point transforms are allowed a small numerical error.
            for actual, expected in zip(self.forward.compose(self.inverse).matrix, AffineTransform.identity().matrix):
                if abs(actual - expected) > 1e-8:
                    raise TransformError("stored inverse does not invert the forward transform")

    @classmethod
    def from_forward(
        cls,
        from_space: str,
        to_space: str,
        source_dimensions: tuple[int, int],
        target_dimensions: tuple[int, int],
        forward: AffineTransform,
        operation_revision: int | str | None = None,
    ) -> "CoordinateTransform":
        return cls(from_space, to_space, source_dimensions, target_dimensions, forward, forward.inverse(), operation_revision)

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        return self.forward.apply_point(x, y)

    def unmap_point(self, x: float, y: float) -> tuple[float, float]:
        return self.inverse.apply_point(x, y)

    def to_dict(self) -> dict[str, object]:
        return {
            "fromSpace": self.from_space,
            "toSpace": self.to_space,
            "sourceDimensions": list(self.source_dimensions),
            "targetDimensions": list(self.target_dimensions),
            "forward": self.forward.to_dict(),
            "inverse": self.inverse.to_dict(),
            "operationRevision": self.operation_revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CoordinateTransform":
        required = {"fromSpace", "toSpace", "sourceDimensions", "targetDimensions", "forward", "inverse", "operationRevision"}
        if set(value) != required:
            raise TransformError("coordinate transform has unknown or missing fields")
        return cls(
            str(value["fromSpace"]),
            str(value["toSpace"]),
            tuple(int(v) for v in value["sourceDimensions"]),
            tuple(int(v) for v in value["targetDimensions"]),
            AffineTransform.from_dict(value["forward"]),
            AffineTransform.from_dict(value["inverse"]),
            value["operationRevision"],
        )


Transform = AffineTransform
TransformRecord = CoordinateTransform


def pixel_center(index: int) -> float:
    if not isinstance(index, int) or index < 0:
        raise TransformError("pixel index must be a non-negative integer")
    return index + 0.5


def map_pixel_center(index: int, origin: int, scale: float) -> float:
    if scale <= 0 or not isfinite(scale):
        raise TransformError("pixel scale must be positive")
    return (pixel_center(index) - origin) * scale - 0.5


def unmap_pixel_center(index: int, origin: int, scale: float) -> float:
    if scale <= 0 or not isfinite(scale):
        raise TransformError("pixel scale must be positive")
    return origin - 0.5 + (pixel_center(index) / scale)
