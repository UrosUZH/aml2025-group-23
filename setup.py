import subprocess
import sys
import gdown
import zipfile
from pathlib import Path

def install_requirements(requirements_file="requirements.txt"):
    print(f"📦 Installing requirements from {requirements_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])

def download_zip(url, output_path):

    if Path(output_path).exists():
        print(f"✅ Zip file already exists: {output_path}. Skipping download.")
        return
    print(f"🌐 Downloading zip file to {output_path}...")
    gdown.download(url=url, output=output_path, quiet=False)

def unzip_file(zip_path, extract_to="."):
    extract_to_path = Path(extract_to)
    if extract_to_path.exists() and any(extract_to_path.iterdir()):
        print(f"✅ Folder already exists and is not empty: {extract_to}. Skipping extraction.")
        return
    print(f"🗂️ Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete.")

if __name__ == "__main__":
    install_requirements("requirements.txt")
    zip_output = "val_rgb_front_clips.zip"
    download_zip(
        url="https://drive.google.com/uc?id=1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG",
        output_path=zip_output
    )
    unzip_file(zip_output, extract_to="data/val_rgb_front_clips")
