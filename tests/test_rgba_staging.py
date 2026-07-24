import sys
import os
import io
import asyncio
from types import SimpleNamespace
from PIL import Image
import numpy as np

# Pre-mock args_manager to avoid argparse conflicts during test execution
fake_args = SimpleNamespace(
    colab=False,
    preset="",
    output_path="",
    temp_path="",
    skip_model_load=True,
    disable_metadata=True,
)
sys.modules["args_manager"] = SimpleNamespace(
    args=fake_args,
    args_parser=SimpleNamespace(args=fake_args, parser=SimpleNamespace()),
)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_rgba_staging():
    # Mock modules.config and modules.util
    import modules.config
    import modules.util
    staging_dir = os.path.join(modules.config.path_outputs, "staging")
    os.makedirs(staging_dir, exist_ok=True)

    from modules.staging_api import upload_staging_image
    from fastapi import UploadFile
    
    # Create a dummy RGBA image
    rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    img_byte_arr = io.BytesIO()
    rgba_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    class MockFile:
        def __init__(self, content):
            self.content = content
        async def read(self):
            return self.content
            
    mock_upload = UploadFile(filename="test.png", file=img_byte_arr)
    
    # Run the upload function (mocking the FastAPI environment)
    # Since we changed it to .convert("RGBA"), it should preserve the alpha channel.
    import modules.staging_api
    # We need to manually set up the request context if needed, but here we call it directly
    response = asyncio.run(upload_staging_image(file=mock_upload))
    
    import json
    data = json.loads(response.body)
    filename = data['file']
    filepath = os.path.join(staging_dir, filename)
    
    with Image.open(filepath) as saved_img:
        print(f"Saved Image Mode: {saved_img.mode}")
        if saved_img.mode == 'RGBA':
            print("RGBA Preservation Test Passed!")
        else:
            print(f"RGBA Preservation Test Failed! Mode is {saved_img.mode}")
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)

if __name__ == '__main__':
    test_rgba_staging()
