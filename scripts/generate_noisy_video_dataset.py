import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

def add_gaussian_noise(img, sigma=25):
    """Add Gaussian noise to image.

    Args:
        img (numpy array): Input image (BGR), range [0, 255].
        sigma (int): Noise level.

    Returns:
        numpy array: Noisy image.
    """
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def generate_noisy_dataset(input_root, output_root, sigma=25):
    """Generate noisy version of a dataset.

    Args:
        input_root (str): Path to clean videos (e.g., train_sharp).
        output_root (str): Path to save noisy videos (e.g., train_noisy).
        sigma (int): Noise intensity.
    """
    if not os.path.exists(input_root):
        print(f"Error: Input root '{input_root}' does not exist.")
        return

    # Get all clips
    clips = sorted(glob.glob(os.path.join(input_root, '*')))
    print(f"Found {len(clips)} clips in {input_root}")

    for clip_path in tqdm(clips):
        if not os.path.isdir(clip_path):
            continue

        clip_name = os.path.basename(clip_path)
        save_dir = os.path.join(output_root, clip_name)
        os.makedirs(save_dir, exist_ok=True)

        # Process each frame
        frames = sorted(glob.glob(os.path.join(clip_path, '*.png')))
        for frame_path in frames:
            frame_name = os.path.basename(frame_path)
            save_path = os.path.join(save_dir, frame_name)

            # Read, Add Noise, Save
            img = cv2.imread(frame_path)
            if img is None:
                continue

            noisy_img = add_gaussian_noise(img, sigma=sigma)
            cv2.imwrite(save_path, noisy_img)

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Defines where your CLEAN data is (e.g., REDS train_sharp)
    clean_data_path = 'datasets/REDS/train_sharp'

    # Defines where to save the NOISY data
    noisy_data_path = 'datasets/REDS/train_noisy'

    # Noise Level (Standard values: 15, 25, 50)
    NOISE_LEVEL = 25
    # ---------------------

    print(f"Generating noisy dataset from {clean_data_path}...")
    generate_noisy_dataset(clean_data_path, noisy_data_path, sigma=NOISE_LEVEL)
    print("Done! You can now use this path as 'dataroot_lq' in your config with scale: 1.")
