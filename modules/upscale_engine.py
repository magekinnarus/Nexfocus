import torch
import numpy as np
from typing import Callable, List, Tuple
from dataclasses import dataclass
import modules.core as core
import gc
import time
import backend.resources as resources
import backend.utils as backend_utils


_GIB = 1024 ** 3
_HEAVY_TRANSFORMER_ARCHITECTURES = ("HAT", "SWINIR", "DAT")

@dataclass
class Segment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

def split_into_segments(length: int, tile_size: int, overlap: int) -> List[Segment]:
    """Return the windows used by the standard single-overlap traversal."""
    if length <= tile_size:
        return [Segment(0, length)]

    step = max(1, tile_size - overlap)
    positions = list(range(0, length - overlap, step))
    return [Segment(pos, min(length, pos + tile_size)) for pos in positions]


def select_best_tile_size(width: int, height: int, max_safe_tile: int, overlap: int = 32) -> Tuple[int, int, int]:
    """Choose the safe tile with the least real splitter work.

    Ties prefer the larger tile so transformer upscalers pay less per-call
    dispatch overhead. Counts come from the same splitter used by inference.
    """
    floor = 256
    ceil = max(floor, min(1024, int(max_safe_tile)))
    candidates = []
    for tile_size in range(floor, ceil + 1, 64):
        x_count = len(split_into_segments(int(width), tile_size, overlap))
        y_count = len(split_into_segments(int(height), tile_size, overlap))
        total_calls = x_count * y_count
        processed_area = total_calls * tile_size * tile_size
        candidates.append((processed_area, -tile_size, tile_size, x_count, y_count))

    if not candidates:
        x_count = len(split_into_segments(int(width), floor, overlap))
        y_count = len(split_into_segments(int(height), floor, overlap))
        return floor, x_count, y_count

    _, _, tile_size, x_count, y_count = min(candidates)
    return tile_size, x_count, y_count


def cap_tile_for_hardware(
    max_safe_tile: int,
    *,
    total_vram: int,
    model_params: int,
    architecture_id: str | None,
) -> int:
    """Cap tiles that fit nominal VRAM but trigger WDDM shared-memory paging.

    Transformer upscalers have activation working sets that are not represented
    well by parameter count alone.  On low-VRAM Windows devices an oversized
    tile can therefore run without a CUDA OOM while spilling into system RAM.
    """
    arch = (architecture_id or "").upper()
    is_heavy = model_params >= 30_000_000 or any(name in arch for name in _HEAVY_TRANSFORMER_ARCHITECTURES)
    if not is_heavy:
        return int(max_safe_tile)

    if total_vram <= 4 * _GIB:
        practical_cap = 256
    elif total_vram <= 6 * _GIB:
        practical_cap = 320
    elif total_vram <= 8 * _GIB:
        practical_cap = 384
    else:
        practical_cap = 512
    return min(int(max_safe_tile), practical_cap)

class NexUpscaleEngine:
    def __init__(self):
        pass

    def _process_tiled(self, in_img, upscale_fn, scale, tile_size, overlap, device, dtype, output_width, output_height):
        h, w = in_img.shape[2:]
        x_segs = split_into_segments(w, tile_size, overlap)
        y_segs = split_into_segments(h, tile_size, overlap)

        resources.soft_empty_cache()
        gc.collect()

        class _RowProgress:
            def __init__(self, columns: int, rows: int):
                self.columns = columns
                self.rows = rows
                self.completed = 0
                self.row_start = time.time()

            def update(self, amount: int):
                self.completed += amount
                if self.completed % self.columns == 0:
                    row = self.completed // self.columns
                    print(f'[Nex-Engine] Row {row}/{self.rows} completed in {time.time() - self.row_start:.2f}s')
                    self.row_start = time.time()

        # Blend and accumulate on CPU.  Keeping only the active model tile on
        # CUDA prevents Windows from silently paging CUDA memory into system RAM.
        final_output = backend_utils.tiled_scale(
            in_img,
            upscale_fn,
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=overlap,
            upscale_amount=scale,
            out_channels=3,
            output_device='cpu',
            pbar=_RowProgress(len(x_segs), len(y_segs)),
        )

        resources.soft_empty_cache(force=True)
        gc.collect()
        print('[Nex-Engine] Tiled Upscale Completed.')
        return final_output

    @torch.inference_mode()
    def process(self, img: np.ndarray, upscale_fn: Callable[[torch.Tensor], torch.Tensor], 
                scale: int, device: torch.device, is_bgr: bool = True, dtype: torch.dtype = None,
                model_params: int | None = None, architecture_id: str | None = None):
        
        m_dtype = dtype or torch.float32
        resources.begin_memory_phase('upscale', notes={'scale': scale, 'device': str(device)})

        try:
            # Ensure correct CUDA device context
            if device.type == 'cuda' and device.index is not None:
                if torch.cuda.current_device() != device.index:
                    torch.cuda.set_device(device)

            # Input to Tensor
            in_img = core.numpy_to_pytorch(img).movedim(-1, -3).to(device, dtype=m_dtype)
            if is_bgr:
                in_img = in_img[:, [2, 1, 0], :, :]

            h, w = in_img.shape[2:]
            oh, ow = h * scale, w * scale

            # Optimized Hardware-Aware Tiling
            tile_size = 512
            overlap = 32

            if 'cuda' in device.type:
                try:
                    # Enable CuDNN benchmark for consistent tiled shapes
                    torch.backends.cudnn.benchmark = True

                    free, total = torch.cuda.mem_get_info(device)
                    usable_vram = free * 0.45

                    # Heavy models need larger safety margins than ESRGAN-lite style models.
                    resolved_model_params = int(model_params or (16 * 1024 * 1024))

                    safety_multiplier = max(800, int(resolved_model_params / 53248))
                    pixel_size = 2 if m_dtype == torch.float16 else 4
                    est_pixels = usable_vram / (pixel_size * safety_multiplier)
                    max_safe_tile = int(est_pixels ** 0.5)
                    max_safe_tile = cap_tile_for_hardware(
                        max_safe_tile,
                        total_vram=total,
                        model_params=resolved_model_params,
                        architecture_id=architecture_id,
                    )

                    tile_size, x_count, y_count = select_best_tile_size(w, h, max_safe_tile, overlap)
                    arch_label = architecture_id or 'unknown'
                    print(
                        f'[Nex-Engine] VRAM: {free//1024**2}MB | Best-Fit Tile: {tile_size} '
                        f'| Grid: {x_count}x{y_count} ({x_count * y_count} calls) '
                        f'| Precision: {m_dtype} | Architecture: {arch_label} '
                        f'| Parameters: {resolved_model_params}'
                    )
                except Exception:
                    pass

            if h <= tile_size and w <= tile_size:
                start_t = time.time()
                out_img = upscale_fn(in_img)
                print(f'[Nex-Engine] Full image processed in {time.time() - start_t:.2f}s')
            else:
                attempted_tile = tile_size
                while True:
                    try:
                        out_img = self._process_tiled(
                            in_img,
                            upscale_fn,
                            scale,
                            attempted_tile,
                            overlap,
                            device,
                            m_dtype,
                            ow,
                            oh,
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        if device.type != 'cuda' or attempted_tile <= 256:
                            raise
                        next_tile = max(256, attempted_tile - 64)
                        print(
                            f'[Nex-Engine] CUDA OOM at tile {attempted_tile}; '
                            f'retrying with tile {next_tile}.'
                        )
                        attempted_tile = next_tile
                        resources.soft_empty_cache(force=True)
                        gc.collect()

            if is_bgr:
                out_img = out_img[:, [2, 1, 0], :, :]

            out_img = torch.clamp(out_img.movedim(-3, -1), 0, 1)
            return core.pytorch_to_numpy(out_img)[0]
        finally:
            resources.end_memory_phase('upscale', notes={'completed': True, 'scale': scale})
