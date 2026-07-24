import os
import sys
import types
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_args = types.SimpleNamespace(
    colab=False,
    preset=None,
    output_path=None,
    temp_path=None,
    skip_model_load=False,
    disable_preset_selection=False,
    disable_image_log=False,
)

sys.modules['args_manager'] = types.ModuleType('args_manager')
sys.modules['args_manager'].args = mock_args

import modules.pipeline.tiled_refinement as tiled

def test_select_resolution():
    print("Testing select_tile_resolution...")
    # 1:1 image
    res, nx, ny, overlap_w, overlap_h = tiled.select_tile_resolution(1024, 1024)
    assert res == (1024, 1024)
    
    # Wide image (e.g. 21:9)
    res, nx, ny, overlap_w, overlap_h = tiled.select_tile_resolution(2048, 873)
    # Target ratio ~2.34
    print(f"Matched {2048/873:.2f} to {res[0]/res[1]:.2f} ({res[0]}x{res[1]})")
    assert res[0] > res[1]

def test_split_and_stitch():
    print("Testing split and stitch logic...")
    # Create dummy image 2048x1024
    image = np.zeros((1024, 2048, 3), dtype=np.uint8)
    image[:, :1024, 0] = 255 # Red left
    image[:, 1024:, 1] = 255 # Green right
    
    tile_w, tile_h = 1024, 1024
    nx, ny = 2, 1
    overlap_w, overlap_h = 0, 0
    tiles = tiled.split_into_tiles(image, tile_w, tile_h, nx, ny, overlap_w, overlap_h)
    print(f"Split into {len(tiles)} tiles")
    
    # Mock no-op refinement
    # Just return the same image
    refined_tiles = [t for t in tiles]
    
    stitched = tiled.stitch_tiles(refined_tiles, image.shape, tile_w, tile_h)
    
    # Check dimensions
    assert stitched.shape == image.shape
    
    # Check content consistency (mean absolute error)
    mae = np.mean(np.abs(stitched.astype(float) - image.astype(float)))
    print(f"Stitch MAE: {mae:.4f}")
    assert mae < 1.0 # Should be very close to zero

if __name__ == "__main__":
    try:
        test_select_resolution()
        test_split_and_stitch()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
