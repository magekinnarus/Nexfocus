import os
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()

def get_redirect_url(version_id, token):
    """Handshake with CivitAI to get the actual file URL."""
    api_url = f"https://civitai.com/api/download/models/{version_id}?token={token}"
    
    # We only need the headers to find where the redirect points
    response = requests.head(api_url, allow_redirects=True, timeout=10)
    return response.url

def download_local_model():
    model_name = "revAnimated_v2Pruned.safetensors"
    version_id = "474453"
    download_dir = r"D:\AI\Imagine\models\checkpoints\sd15\base"
    
    civitai_token = os.getenv('CIVITAI_TOKEN', '')

    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)

    print(f"Resolving redirect for ID {version_id}...")
    try:
        # 1. Get the final, direct URL (the one starting with b2.civitai.com)
        direct_url = get_redirect_url(version_id, civitai_token)
        print(f"Target found: {direct_url[:50]}...")

        # 2. Build a minimal aria2c command
        command = [
            'aria2c',
            '--console-log-level=warn',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            '--check-certificate=false',
            '-x', '4',
            '-s', '4',
            '-k', '1M',
            '--dir', download_dir,
            '--out', model_name,
            direct_url # Pass the resolved URL directly
        ]

        # 3. Clean up and run
        aria_meta = os.path.join(download_dir, f"{model_name}.aria2")
        if os.path.exists(aria_meta):
            os.remove(aria_meta)

        subprocess.check_call(command)
        print(f"\n✅ Download complete: {model_name}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    download_local_model()
