import os
import glob
from PIL import Image

def generate_video_meta_info(data_root, meta_info_file):
    """Generate meta info for video datasets.

    Args:
        data_root (str): Path to the dataset root (e.g., 'datasets/MyVideoDataset/train_sharp').
        meta_info_file (str): Path to save the meta info txt file.
    """
    if not os.path.exists(data_root):
        print(f"Error: Data root '{data_root}' does not exist.")
        return

    # Get all subfolders (clips)
    subfolders = sorted(glob.glob(os.path.join(data_root, '*')))

    with open(meta_info_file, 'w') as f:
        for folder_path in subfolders:
            if not os.path.isdir(folder_path):
                continue

            folder_name = os.path.basename(folder_path)

            # Count images
            img_list = sorted(glob.glob(os.path.join(folder_path, '*')))
            if not img_list:
                print(f"Warning: No images found in {folder_name}")
                continue

            num_frames = len(img_list)

            # Read first image to get dimensions
            try:
                with Image.open(img_list[0]) as img:
                    width, height = img.size
                    mode = img.mode
                    if mode == 'RGB':
                        n_channel = 3
                    elif mode == 'L':
                        n_channel = 1
                    else:
                        n_channel = 3 # Default to 3 if unknown
            except Exception as e:
                print(f"Error reading image {img_list[0]}: {e}")
                continue

            # Format: FolderName NumFrames (Height,Width,Channels)
            # Note: BasicSR often expects (Height, Width, Channels)
            info = f'{folder_name} {num_frames} ({height},{width},{n_channel})'
            print(f"Adding: {info}")
            f.write(f'{info}\n')

    print(f"Successfully generated meta info at: {meta_info_file}")

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # 1. Path to your CLEAN/BS videos (Ground Truth)
    #    Example: 'datasets/MyVideoDataset/train_sharp'
    my_data_root = '/content/drive/MyDrive/basicsr/dataset/REDS/val_sharp'

    # 2. Where to save the text file
    #    Example: 'basicsr/data/meta_info/meta_info_MyDataset.txt'
    my_save_path = '/content/BasicSR/basicsr/data/meta_info/meta_info_Video_GT1.txt'
    # ---------------------

    generate_video_meta_info(my_data_root, my_save_path)
