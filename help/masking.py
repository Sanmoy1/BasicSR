import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt


class MaskedNoiseAugmentor:
    """
    Adds spatially varying (partial) noise to images using random masks.
    Supports several mask generation modes for denoising training.
    """

    def __init__(
        self,
        noise_std_range=(1, 25),
        mask_prob_range=(0.3, 0.6),
        mask_mode=['blobs', 'rectangles', 'perlin', 'soft'],  # List of modes to randomly choose from
        use_gaussian_blur=True,
        blend_power=0.7  # Controls gradient falloff: lower = more gradual, higher = sharper center
    ):
        self.noise_std_range = noise_std_range
        self.mask_prob_range = mask_prob_range
        # Ensure mask_mode is a list
        if isinstance(mask_mode, str):
            self.mask_mode = [mask_mode]
        else:
            self.mask_mode = mask_mode
        self.use_gaussian_blur = use_gaussian_blur
        self.blend_power = blend_power

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: torch.Tensor (C, H, W), float32, range (0, 1)
        Returns:
            noisy_img: torch.Tensor (C, H, W)
        """
        img = img.clone()
        _, H, W = img.shape

        # 1. Generate random binary/soft mask
        mask = self._generate_mask(H, W)
        mask = torch.from_numpy(mask).float().to(img.device)

        # 2. Apply distance-based smooth blending from center
        if self.use_gaussian_blur:
            mask_np = mask.cpu().numpy().astype(np.uint8)

            # Compute distance transform from the center of masked regions
            # Distance transform gives distance to nearest zero pixel
            dist_transform = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)

            # Normalize distance transform to [0, 1]
            if dist_transform.max() > 0:
                dist_transform = dist_transform / dist_transform.max()

            # Apply Gaussian blur for additional smoothing
            dist_transform = cv2.GaussianBlur(dist_transform, (21, 21), sigmaX=7)

            # Normalize again to ensure [0, 1] range
            if dist_transform.max() > 0:
                dist_transform = (dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min() + 1e-8)

            # Apply power function to control falloff (higher power = sharper center, softer edges)
            # You can adjust this value: 0.5 = more gradual, 2.0 = sharper center
            dist_transform = np.power(dist_transform, self.blend_power)

            mask = torch.from_numpy(dist_transform.astype(np.float32)).to(img.device)

        return mask

    def _generate_mask(self, H, W):
        # Select random mode
        mode = random.choice(self.mask_mode)

        # Generate initial mask
        if mode == 'blobs':
            Mask = self._blob_mask(H, W)
        elif mode == 'rectangles':
            Mask = self._rect_mask(H, W)
        elif mode == 'perlin':
            Mask = self._perlin_mask(H, W)
        elif mode == 'soft':
            Mask = self._soft_mask(H, W)
        else:
            raise ValueError(f"Unknown mask_mode '{mode}'")

        # Convert to torch tensor
        Mask = torch.from_numpy(Mask).float()

        # Calculate initial black coverage (percentage of 0.0-valued pixels)
        black_coverage = torch.sum(Mask == 0.0).item() / (H * W)

        # If coverage is less than target (e.g., 70%), iteratively add more masks
        iterations = 0
        max_iterations = 2
        target_coverage = 0.3  # black region target proportion

        while black_coverage > target_coverage and iterations < max_iterations:
            # Generate additional mask using the same mode
            if mode == 'blobs':
                extra_Mask = self._blob_mask(H, W)
            elif mode == 'rectangles':
                extra_Mask = self._rect_mask(H, W)
            elif mode == 'perlin':
                extra_Mask = self._perlin_mask(H, W)
            elif mode == 'soft':
                extra_Mask = self._soft_mask(H, W)

            # Convert to torch tensor
            extra_Mask = torch.from_numpy(extra_Mask).float()

            # Add the extra mask and clamp to [0, 1]
            Mask = torch.clamp(Mask + extra_Mask, 0, 1)

            # Recalculate black coverage
            black_coverage = torch.sum(Mask == 0.0).item() / (H * W)
            print(f"  Black coverage: {black_coverage:.2f}")
            iterations += 1

        # Convert back to numpy array for compatibility with existing code
        return Mask.numpy()


    def _blob_mask(self, H, W):
        """Random binary blobs."""
        prob = random.uniform(*self.mask_prob_range)
        mask = np.random.rand(H, W)
        mask = (mask > prob).astype(np.float32)
        return mask

    def _rect_mask(self, H, W):
        """Random rectangular regions."""
        mask = np.zeros((H, W), np.float32)
        n_rects = random.randint(3, 8)
        for _ in range(n_rects):
            x1 = random.randint(0, W - 1)
            y1 = random.randint(0, H - 1)
            x2 = min(W, x1 + random.randint(W // 8, W // 3))
            y2 = min(H, y1 + random.randint(H // 8, H // 3))
            mask[y1:y2, x1:x2] = 1.0
        return mask

    def _perlin_mask(self, H, W, scale=32):
        """Perlin-like smooth noise mask (approximation)."""
        grid = np.random.rand(H // scale + 1, W // scale + 1)
        mask = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask > 0.5).astype(np.float32)
        return mask

    def _soft_mask(self, H, W):
        """Soft random mask (gradual transitions)."""
        mask = np.random.rand(H, W).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        # Apply binary thresholding to make it 0 or 1
        mask = (mask > 0.5).astype(np.float32)
        return mask




# Load the test image
img_path = "D:/personal projects/samsung/BasicSR/help/test/image/image.jpg"
try:
    img_pil = Image.open(img_path).convert('RGB')
    print(f"Loaded image: {img_path} with size {img_pil.size}")
except FileNotFoundError:
    print(f"Error: '{img_path}' not found in current directory.")
    print("Please ensure 'image.png' exists in the same directory as this script.")
    exit(1)

# Convert to torch tensor (C, H, W) in range [0, 1]
img_np = np.array(img_pil).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

print(f"Image tensor shape: {img_tensor.shape}")

# Test different mask modes
mask_modes = ['blobs', 'rectangles', 'perlin', 'soft']



for idx, mode in enumerate(mask_modes):
    print(f"\nProcessing mask mode: {mode}")

    # Create augmentor with current mask mode
    augmentor = MaskedNoiseAugmentor(
        noise_std_range=(10, 30),
        mask_prob_range=(0.3, 0.7),
        mask_mode=[mode],  # Pass as list with single mode
        use_gaussian_blur=True
    )

    # Get mask from augmentor (now __call__ returns mask directly)
    mask = augmentor(img_tensor)

    # Ensure mask is 2D (H, W)
    if mask.dim() == 3:
        mask = mask[0]  # Take first channel if 3D

    mask_np = mask.cpu().numpy()

    # Convert tensors to numpy for visualization
    original_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Since __call__ now returns mask only, create noisy image manually for visualization
    noise_std = random.uniform(10, 30) / 255.0
    noise = torch.randn_like(img_tensor) * noise_std
    noisy_img = torch.clamp(img_tensor + noise * mask.unsqueeze(0), 0.0, 1.0)
    noisy_np = noisy_img.permute(1, 2, 0).cpu().numpy()

    # Create mask visualization (black=0, white=1)
    mask_vis = (mask_np * 255).astype(np.uint8)
    mask_vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

    # Create masked region visualization (show where noise is applied)
    masked_region = original_np.copy()
    # Overlay red tint where mask == 1
    red_overlay = np.zeros_like(original_np)
    red_overlay[:, :, 0] = mask_np  # Red channel
    masked_region = np.clip(masked_region * 0.6 + red_overlay * 0.4, 0, 1)



    # Print statistics
    noise_percentage = (mask_np.sum() / mask_np.size) * 100
    print(f"  Mask coverage: {noise_percentage:.2f}% of pixels will have noise")

plt.tight_layout()

# Save the visualization
output_path = r'D:\personal projects\samsung\BasicSR\help\test\masking_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Also save individual mask visualizations
print("\nSaving individual mask visualizations...")
for mode in mask_modes:
    augmentor = MaskedNoiseAugmentor(
        noise_std_range=(10, 30),
        mask_prob_range=(0.3, 0.7),
        mask_mode=[mode],  # Pass as list with single mode
        use_gaussian_blur=True
    )

    # Get mask from augmentor
    mask = augmentor(img_tensor)

    # Ensure mask is 2D
    if mask.dim() == 3:
        mask = mask[0]


    mask_vis = (mask.cpu().numpy() * 255).astype(np.uint8)
    output_path = rf'D:\personal projects\samsung\BasicSR\help\test\mask_{mode}.png'
    cv2.imwrite(output_path, mask_vis)
    print(f"  Saved: {output_path}")

plt.show()
print("\nDone! Check the visualization window and saved images.")