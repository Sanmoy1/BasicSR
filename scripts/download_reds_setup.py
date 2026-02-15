import os
import requests
import zipfile
import shutil

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    # URL for a small subset of REDS (REDS4) or a placeholder sample
    # Since I cannot browse the live web for a direct link effectively without verification,
    # I will provide instructions and a placeholder URL.
    # USER: Please replace this URL with a valid REDS4 sample link if you have one.
    # Otherwise, this script attempts to download a sample from a common repo if available.

    # Using a known sample link for REDS4 (Validation Set - Sharp & Sharp_Bicubic)
    # Note: These are large files. If this fails, user must download manually.

    print("NOTE: The full REDS dataset is very large.")
    print("This script will help you organize folders if you download the dataset manually.")
    print("   Download Link: https://seungjunnah.github.io/Datasets/reds.html")

    # Basic Directory Setup
    root_dir = 'datasets/REDS'
    os.makedirs(os.path.join(root_dir, 'train_sharp'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train_sharp_bicubic/X4'), exist_ok=True)

    print(f"\nCreated directory structure at: {root_dir}")
    print("1. Download 'train_sharp' and 'train_sharp_bicubic' from the official site.")
    print(f"2. Extract clean videos to: {os.path.join(root_dir, 'train_sharp')}")
    print(f"3. Extract noisy videos to: {os.path.join(root_dir, 'train_sharp_bicubic/X4')}")
    print("\nFor a quick test, you can create dummy images:")

    create_dummy = input("Do you want to create a dummy dataset for testing code? (y/n): ")
    if create_dummy.lower() == 'y':
        create_dummy_dataset(root_dir)

def create_dummy_dataset(root_dir):
    from PIL import Image
    import numpy as np

    print("Creating dummy dataset...")

    # Create 2 clips, 15 frames each
    clips = ['000', '001']
    num_frames = 15
    width, height = 1280, 720

    for clip in clips:
        clean_dir = os.path.join(root_dir, 'train_sharp', clip)
        noisy_dir = os.path.join(root_dir, 'val_sharp_bicubic/X4', clip)
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        for i in range(num_frames):
            frame_name = f"{i:08d}.png"

            # Random uniform noise for "clean" (just to have content)
            img_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            Image.fromarray(img_np).save(os.path.join(clean_dir, frame_name))

            # Add more noise for "noisy"
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            noisy_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            Image.fromarray(noisy_np).save(os.path.join(noisy_dir, frame_name))

    print("Dummy dataset created successfully!")
    print("You can now run 'python scripts/generate_meta_info_video.py' to generate the meta info.")

if __name__ == '__main__':
    main()
