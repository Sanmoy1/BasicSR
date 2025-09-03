# Hybrid Dataset Denoising Pipeline

## 1. Overview

This document describes a comprehensive image denoising pipeline that leverages both paired and unpaired data for training. The pipeline combines the strengths of supervised learning (using paired noisy-clean images) with the data diversity advantages of unpaired data (using only clean images with synthetic degradation).

### Key Components

- **AICNet Architecture**: A U-Net style encoder-decoder network with 4 downsampling and 4 upsampling layers
- **Hybrid Dataset**: Combines paired data (LQ-GT pairs) with unpaired data (GT-only)
- **Synthetic Degradation**: Realistic noise simulation for unpaired data
- **Multi-Loss Training**: Combination of pixel, perceptual, frequency, and structural losses

## 2. AICNet Architecture

The AICNet architecture follows an encoder-decoder structure optimized for image denoising:

```
AICNet
├── Encoder (4 downsampling layers)
└── Decoder (4 upsampling layers with skip connections)
```

### Key Features

- **Encoder**: Progressively reduces spatial dimensions while increasing feature depth
- **Decoder**: Gradually recovers spatial details through upsampling
- **Skip Connections**: Preserves high-frequency details from encoder to decoder
- **Input/Output**: Works with RGB images (3 channels)

## 3. Hybrid Dataset Implementation

### RealESRGANHybridDataset

The `RealESRGANHybridDataset` class combines paired and unpaired data sources:

```python
class RealESRGANHybridDataset(data.Dataset):
    """Hybrid dataset that combines paired and unpaired data."""
    
    def __init__(self, opt):
        super(RealESRGANHybridDataset, self).__init__()
        # Initialize with both paired and unpaired data sources
        # Set consistent crop size (gt_size) for both data types
        self.gt_size = opt.get('gt_size', 400)
        
    def __getitem__(self, index):
        path_info = self.paths[index]
        if path_info['gt_path'] == path_info['lq_path']:
            return self._load_unpaired_data(path_info)
        else:
            return self._load_paired_data(path_info)
```

### Data Loading Process

1. **Paired Data**:
   - Load GT and LQ images directly
   - Apply consistent cropping (400×400 pixels)
   - Apply augmentation (horizontal flip, rotation)
   - Convert to tensors

2. **Unpaired Data**:
   - Load GT image only
   - Apply augmentation and cropping
   - Generate synthetic degradation kernels
   - Return GT image with degradation parameters

## 4. Synthetic Degradation Pipeline

For unpaired data, a two-stage degradation process is applied:

1. **First Degradation Stage**:
   - Blur: Apply random blur kernel (isotropic/anisotropic Gaussian, plateau-shaped, etc.)
   - Resize: Downsample with random interpolation
   - Noise: Add random noise (Gaussian, Poisson, etc.)
   - JPEG: Apply random JPEG compression

2. **Second Degradation Stage**:
   - Blur: Apply another random blur kernel
   - Resize: Upsample to target size
   - Noise: Add additional noise
   - JPEG: Apply final JPEG compression

## 5. Model Implementation

### RealESRGANModel with Hybrid Data Support

The model is extended to handle both paired and unpaired data:

```python
def feed_data_paired_unpaired(self, data):
    """Handle mixed batches of paired and unpaired data."""
    # For paired data
    if 'gt' in data and 'lq' in data and data['gt'].shape == data['lq'].shape:
        self.gt = data['gt'].to(self.device)
        self.lq = data['lq'].to(self.device)
    # For unpaired data
    elif 'gt' in data:
        self.gt = data['gt'].to(self.device)
        # Apply synthetic degradation
        self.lq = self.degrade_process(self.gt)
    else:
        raise ValueError('Invalid data format')
```

## 6. Loss Functions

The training employs multiple loss functions to achieve high-quality denoising:

### Pixel-wise Losses

- **Charbonnier Loss**: A differentiable variant of L1 loss that better handles outliers
  ```
  L_char(x, y) = √((x - y)² + ε²)
  ```

### Perceptual Losses

- **VGG Perceptual Loss**: Measures differences in feature space using a pre-trained VGG network
  ```
  L_percep = Σ w_i * ||φ_i(x) - φ_i(y)||_1
  ```

### Frequency Domain Losses

- **FFT Loss**: Penalizes differences in frequency domain to preserve textures
  ```
  L_fft = ||FFT(x) - FFT(y)||_p
  ```

### Structural Losses

- **SSIM Loss**: Focuses on preserving structural information
  ```
  L_ssim = 1 - SSIM(x, y)
  ```

## 7. Training Process

1. **Data Preparation**:
   - Load mixed batch of paired and unpaired data
   - Apply consistent cropping to ensure uniform tensor sizes

2. **Forward Pass**:
   - For paired data: Use LQ directly as input
   - For unpaired data: Generate synthetic LQ from GT

3. **Loss Calculation**:
   - Calculate combined loss using weighted sum of all loss components
   - Backpropagate and update model parameters

4. **Validation**:
   - Periodically evaluate on validation set
   - Track metrics like PSNR, SSIM, and LPIPS

## 8. Potential Upgrades

### Denoised Consistency Supervision (DCS)

DCS can be integrated to improve texture preservation:

1. **Modify AICNet** to predict both denoised image and noise
2. **Create renoised variants** by adding scaled predicted noise
3. **Enforce consistency** across denoised variants

### Other Potential Improvements

- **Attention Mechanisms**: Incorporate channel or spatial attention
- **Progressive Training**: Train with gradually increasing degradation severity
- **Curriculum Learning**: Start with easier samples and progress to harder ones

## 9. Conclusion

This hybrid dataset denoising pipeline combines the best of supervised and unsupervised approaches. By using both paired and unpaired data with consistent processing, it overcomes the limitations of each approach individually:

- **Paired data** provides direct supervision for realistic degradations
- **Unpaired data** with synthetic degradation increases training diversity
- **Consistent tensor sizes** enable efficient batch processing
- **Multi-component loss** ensures both fidelity and perceptual quality

The pipeline is flexible and can be extended with additional components like DCS for further improvements in denoising performance while preserving natural textures.