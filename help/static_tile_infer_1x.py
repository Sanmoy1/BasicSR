import argparse
import cv2
import glob
import math
import numpy as np
import os
import torch
from torch.nn import functional as F

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from basicsr.metrics import calculate_psnr, calculate_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default='datasets/Set5/LRbicx2',
        help='Input directory')
    parser.add_argument(
        '--output',
        type=str,
        default='results/static_tile_infer_1x',
        help='Output directory')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RRDBNet_x1.pth',
        help='Path to the model')
    parser.add_argument(
        '--tile_size',
        type=int,
        default=512,
        help='Tile size for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.output, exist_ok=True)

    # ----------------------------------------
    # 1. Load Model
    # ----------------------------------------
    # RRDBNet(num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    # User requested 1x scale. Assuming standard params for others.
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=1, num_feat=64, num_block=23, num_grow_ch=32)

    # Load weights
    if os.path.exists(args.model_path):
        loadnet = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        print(f'Model loaded from {args.model_path}')
    else:
        print(f'Warning: Model path {args.model_path} does not exist. Initializing with random weights for testing.')

    model.eval()
    model = model.to(device)

    # ----------------------------------------
    # 2. Process Images
    # ----------------------------------------
    img_list = sorted(glob.glob(os.path.join(args.input, '*')))
    print(f'Found {len(img_list)} images in {args.input}')

    # For table formatting
    print(f"{'Image Name':<30} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 56)

    for img_path in img_list:
        img_name = os.path.basename(img_path)

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_t = img2tensor(img, bgr2rgb=True, float32=True)
        img_t = img_t.unsqueeze(0).to(device) # (1, 3, H, W)

        batch, channel, height, width = img_t.shape
        output_t = torch.zeros_like(img_t) # Same shape for 1x scale

        # ----------------------------------------
        # 3. Tiling Logic
        # ----------------------------------------
        tile_size = args.tile_size

        # Number of tiles
        h_tiles = math.ceil(height / tile_size)
        w_tiles = math.ceil(width / tile_size)

        tile_count = 0

        print(f'\nProcessing {img_name}: {height}x{width}')

        with torch.no_grad():
            for i in range(h_tiles):
                for j in range(w_tiles):
                    tile_count += 1

                    # Determine coordinates
                    h_start = i * tile_size
                    h_end = min(h_start + tile_size, height)
                    w_start = j * tile_size
                    w_end = min(w_start + tile_size, width)

                    # Because we want fixed 512 input if possible, but boundaries might be smaller.
                    # The user said "give input to the model of 512 tile images".
                    # However, if the image isn't divisible, the last tile will be smaller.
                    # Standard practice: crop the exact region.
                    # If strictly 512 is required, padding would be needed, but user said "padding will be zero".
                    # Implementation: Crop actual region. If model accepts variable size (like RRDBNet), this is fine.

                    input_tile = img_t[:, :, h_start:h_end, w_start:w_end]

                    print(f'  Tile {tile_count}: Shape {input_tile.shape}')

                    # Inference
                    output_tile = model(input_tile)

                    # Fill output
                    output_t[:, :, h_start:h_end, w_start:w_end] = output_tile

        # ----------------------------------------
        # 4. Metrics & Saving
        # ----------------------------------------
        # Tensor to Img
        output_img = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1)) # uint8 0-255

        # Get Reference Image (Input) as uint8 for comparison
        # (We reload or convert original)
        input_img_uint8 = tensor2img(img_t, rgb2bgr=True, min_max=(0, 1))

        # Metrics
        # Note: crop_border=0 because 0 padding was used and assumed no artifacts?
        # Usually we crop border, but user didn't specify. Assuming 0 border for now or standard 4.
        # Let's use 0 to be consistent with "0 padding" instruction logic.

        psnr_val = calculate_psnr(output_img, input_img_uint8, crop_border=0, test_y_channel=True)
        ssim_val = calculate_ssim(output_img, input_img_uint8, crop_border=0, test_y_channel=True)

        print(f"Finished {img_name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

        # Save
        save_path = os.path.join(args.output, img_name)
        cv2.imwrite(save_path, output_img)

        # Add to summary (re-print for visibility in log)
        # We also need to print the matrix mid-way if requested, but printing one by one is effectively that.
        # print(f"{img_name:<30} | {psnr_val:<10.4f} | {ssim_val:<10.4f}")

if __name__ == '__main__':
    main()
