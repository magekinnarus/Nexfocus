"""Deterministic first-pass document rasterization for preview and export."""

from __future__ import annotations

import io
import math
from typing import Callable

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .schema import Document, LayerRecord, ObjectRecord
from .transforms import AffineTransform


class RenderError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


AssetLoader = Callable[[str], bytes]
_RENDERABLE_LAYER_KINDS = {
    "raster", "paint", "vector", "shape", "guide", "candidate", "patch", "proxy",
    "source-composite", "depth", "object",
}


def _rgba(value: object, opacity: float = 1.0) -> tuple[int, int, int, int]:
    if not isinstance(value, str):
        raise RenderError("UNSUPPORTED_STYLE", "render requires a hexadecimal color")
    text = value.strip().lstrip("#")
    if len(text) not in {6, 8}:
        raise RenderError("UNSUPPORTED_STYLE", "render supports #RRGGBB or #RRGGBBAA colors")
    try:
        channels = [int(text[index:index + 2], 16) for index in range(0, len(text), 2)]
    except ValueError as exc:
        raise RenderError("UNSUPPORTED_STYLE", "render color is malformed") from exc
    if len(channels) == 3:
        channels.append(255)
    channels[3] = max(0, min(255, round(channels[3] * opacity)))
    return tuple(channels)  # type: ignore[return-value]


def _transform_points(points: list[tuple[float, float]], transform: AffineTransform) -> list[tuple[float, float]]:
    return [transform.apply_point(x, y) for x, y in points]


def _combined(parent: AffineTransform, layer: LayerRecord, obj: ObjectRecord | None = None) -> AffineTransform:
    result = parent.compose(layer.transform)
    return result if obj is None else result.compose(obj.transform)


def _scale_factor(transform: AffineTransform) -> float:
    return math.sqrt(abs(transform.determinant()))


def _paint_object(target: Image.Image, obj: ObjectRecord, transform: AffineTransform) -> None:
    geometry = obj.geometry
    style = obj.style
    width = max(1, round(float(style.get("width", style.get("strokeWidth", 1))) * _scale_factor(transform)))
    alpha = max(0.0, min(1.0, float(style.get("opacity", 1.0))))
    mode = style.get("mode", "paint")
    paint = Image.new("RGBA", target.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(paint, "RGBA")

    if obj.kind == "raster-placement":
        raise RenderError("RASTER_REQUIRES_ASSET", "raster placement must be rendered by its asset record")
    if obj.kind not in {"shape", "path", "paint-stroke", "guide", "patch"}:
        raise RenderError("UNSUPPORTED_OBJECT_KIND", f"unsupported object kind: {obj.kind}")

    if obj.kind == "shape":
        shape = geometry.get("shape")
        x = float(geometry.get("x", 0))
        y = float(geometry.get("y", 0))
        w = float(geometry.get("width", 0))
        h = float(geometry.get("height", 0))
        if shape == "line":
            points = [(x, y), (x + w, y + h)]
        elif shape == "ellipse":
            points = [
                (x + w / 2 + math.cos(index * math.tau / 64) * w / 2,
                 y + h / 2 + math.sin(index * math.tau / 64) * h / 2)
                for index in range(64)
            ]
        elif shape in {"rectangle", "polygon"}:
            points = geometry.get("points") if shape == "polygon" else [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            if not isinstance(points, list) or len(points) < 2:
                raise RenderError("UNSUPPORTED_GEOMETRY", "shape needs at least two points")
            points = [(float(point[0]), float(point[1])) for point in points]
        else:
            raise RenderError("UNSUPPORTED_OBJECT_KIND", f"unsupported shape: {shape}")
        points = _transform_points(points, transform)
        fill = style.get("fill")
        stroke = style.get("stroke", style.get("color"))
        if fill is not None and shape != "line":
            draw.polygon(points, fill=_rgba(fill, alpha))
        if stroke is not None:
            draw.line(points + ([points[0]] if shape != "line" else []), fill=_rgba(stroke, alpha), width=width, joint="curve")
        target.alpha_composite(paint)
        return

    raw_points = geometry.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise RenderError("UNSUPPORTED_GEOMETRY", "path or stroke needs points")
    points = _transform_points([(float(p[0]), float(p[1])) for p in raw_points], transform)
    closed = bool(geometry.get("closed", False))
    fill = style.get("fill")
    color = _rgba(style.get("color", style.get("stroke", "#000000")), alpha)

    if mode == "erase":
        erase = Image.new("L", target.size, 0)
        erase_draw = ImageDraw.Draw(erase)
        erase_draw.line(points, fill=round(255 * alpha), width=width, joint="curve")
        radius = round((1.0 - max(0.0, min(1.0, float(style.get("hardness", 1.0))))) * width / 4)
        if radius > 0:
            erase = erase.filter(ImageFilter.GaussianBlur(radius))
        target.putalpha(ImageChops.subtract(target.getchannel("A"), erase))
        return

    if closed and fill is not None:
        draw.polygon(points, fill=_rgba(fill, alpha))
    if len(points) > 1:
        if closed:
            points = points + [points[0]]
        draw.line(points, fill=color, width=width, joint="curve")
    elif width > 1:
        px, py = points[0]
        draw.ellipse((px - width / 2, py - width / 2, px + width / 2, py + width / 2), fill=color)

    radius = round((1.0 - max(0.0, min(1.0, float(style.get("hardness", 1.0))))) * width / 4)
    if radius > 0:
        layer_alpha = paint.getchannel("A")
        blurred = layer_alpha.filter(ImageFilter.GaussianBlur(radius))
        paint.putalpha(ImageChops.lighter(layer_alpha, blurred))
    target.alpha_composite(paint)


def _transformed_asset(data: bytes, size: tuple[int, int], transform: AffineTransform, placement: dict) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGBA")
    except Exception as exc:
        raise RenderError("INVALID_IMAGE_ASSET", "raster asset could not be decoded") from exc
    x = float(placement.get("x", 0))
    y = float(placement.get("y", 0))
    width = float(placement.get("width", image.width))
    height = float(placement.get("height", image.height))
    if width <= 0 or height <= 0:
        raise RenderError("UNSUPPORTED_GEOMETRY", "raster placement dimensions must be positive")
    base = AffineTransform((width / image.width, 0, x, 0, height / image.height, y, 0, 0, 1))
    full = transform.compose(base)
    inverse = full.inverse().matrix
    return image.transform(
        size,
        Image.Transform.AFFINE,
        tuple(inverse[:6]),
        resample=Image.Resampling.BICUBIC,
    )


def _apply_opacity(image: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1.0:
        return image
    alpha = image.getchannel("A").point(lambda value: round(value * opacity))
    image.putalpha(alpha)
    return image


def _layer_asset_ids(document: Document, layer: LayerRecord) -> list[str]:
    values = list(layer.asset_ids)
    for object_id in layer.object_ids:
        obj = document.objects[object_id]
        if obj.asset_id is not None and obj.asset_id not in values:
            values.append(obj.asset_id)
    return values


def render_document(document: Document, asset_loader: AssetLoader) -> Image.Image:
    """Render supported visible semantics or raise a typed fidelity failure."""

    document.validate()
    canvas_size = (document.width, document.height)
    identity = AffineTransform.identity()

    def render_layer(layer_id: str, parent_transform: AffineTransform, inherited_opacity: float) -> Image.Image:
        layer = document.layers[layer_id]
        output = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        if not layer.visible or layer.metadata.get("deleted", False):
            return output
        if layer.kind != "group" and layer.kind not in _RENDERABLE_LAYER_KINDS:
            raise RenderError("UNSUPPORTED_LAYER_KIND", f"unsupported visible layer kind: {layer.kind}")
        if layer.blend_mode != "normal" or layer.clipping_refs:
            raise RenderError("UNSUPPORTED_LAYER_SEMANTICS", "layer blend or clipping semantics are unsupported")
        world_transform = parent_transform.compose(layer.transform)
        opacity = inherited_opacity * float(layer.opacity)
        if layer.kind == "group":
            for child_id in layer.child_ids:
                output.alpha_composite(render_layer(child_id, world_transform, 1.0))
            return _apply_opacity(output, opacity)

        for object_id in layer.object_ids:
            obj = document.objects[object_id]
            if obj.kind == "raster-placement":
                if obj.asset_id is None:
                    raise RenderError("MISSING_RASTER_ASSET", "raster placement has no asset")
                bitmap = _transformed_asset(asset_loader(obj.asset_id), canvas_size, _combined(parent_transform, layer, obj), obj.geometry)
                output.alpha_composite(bitmap)
            else:
                _paint_object(output, obj, _combined(parent_transform, layer, obj))

        placed_ids = {document.objects[item].asset_id for item in layer.object_ids}
        for asset_id in layer.asset_ids:
            if asset_id in placed_ids:
                continue
            bitmap = _transformed_asset(asset_loader(asset_id), canvas_size, world_transform, {})
            output.alpha_composite(bitmap)
        return _apply_opacity(output, opacity)

    result = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    for root_id in document.root_layer_ids:
        result.alpha_composite(render_layer(root_id, identity, 1.0))
    return result


def render_layer(document: Document, layer_id: str, asset_loader: AssetLoader) -> Image.Image:
    """Render one raster layer into document registration for selection tools."""

    if layer_id not in document.layers:
        raise RenderError("UNKNOWN_LAYER", "layer does not exist")
    layer = document.layers[layer_id]
    if layer.kind != "raster":
        raise RenderError("UNSUPPORTED_SELECTION_SOURCE", "selection source must be a raster layer")
    if layer.blend_mode != "normal" or layer.clipping_refs:
        raise RenderError("UNSUPPORTED_LAYER_SEMANTICS", "selection source uses unsupported blend or clipping")
    canvas = Image.new("RGBA", (document.width, document.height), (0, 0, 0, 0))
    parent = AffineTransform.identity()
    inherited_opacity = 1.0
    ancestors: list[LayerRecord] = []
    current = layer
    while current.parent_id is not None:
        current = document.layers[current.parent_id]
        ancestors.append(current)
    for ancestor in reversed(ancestors):
        if not ancestor.visible or ancestor.metadata.get("deleted", False):
            raise RenderError("UNSUPPORTED_SELECTION_SOURCE", "selection source is hidden by an ancestor group")
        if ancestor.blend_mode != "normal" or ancestor.clipping_refs:
            raise RenderError("UNSUPPORTED_LAYER_SEMANTICS", "selection source ancestor uses unsupported blend or clipping")
        parent = parent.compose(ancestor.transform)
        inherited_opacity *= float(ancestor.opacity)
    world = parent.compose(layer.transform)
    if layer.kind == "group":
        raise RenderError("UNSUPPORTED_SELECTION_SOURCE", "group selections require an exact depth composite")
    raster_found = False
    for object_id in layer.object_ids:
        obj = document.objects[object_id]
        if obj.kind == "raster-placement":
            if obj.asset_id is None:
                continue
            canvas.alpha_composite(_transformed_asset(asset_loader(obj.asset_id), canvas.size, _combined(parent, layer, obj), obj.geometry))
            raster_found = True
    for asset_id in layer.asset_ids:
        if not raster_found:
            canvas.alpha_composite(_transformed_asset(asset_loader(asset_id), canvas.size, world, {}))
            raster_found = True
    if not raster_found:
        raise RenderError("UNSUPPORTED_SELECTION_SOURCE", "selection source must contain a raster asset")
    return _apply_opacity(canvas, inherited_opacity * float(layer.opacity))


def render_document_png(document: Document, asset_loader: AssetLoader) -> bytes:
    image = render_document(document, asset_loader)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def image_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()
