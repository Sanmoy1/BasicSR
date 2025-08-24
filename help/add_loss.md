# Add New Loss


Let me check if SSIM loss is already implemented in BasicSR, and if not, how we can add it.

Let me check if SSIM loss is already implemented in BasicSR, and if not, how we can add it.

Let me check if SSIM is implemented as a metric in BasicSR, as we might be able to adapt it for a loss function:

Let me check the SSIM implementation in the metrics module to understand how we can adapt it for a loss function:



# Adding SSIM Loss to BasicSR for RealESRGAN

I see you want to add SSIM loss to your RealESRGAN model. BasicSR has SSIM as a metric but not as a loss function. Let's implement it:

# Adding SSIM Loss to BasicSR for RealESRGAN

I'll show you how to implement SSIM loss for your RealESRGAN model in BasicSR. This requires:

1. Creating a new SSIM loss class
2. Configuring it in your YAML file
3. Integrating it with other losses

## Step 1: Create SSIM Loss Class

Create a new file `basicsr/losses/ssim_loss.py` with this implementation:



## Step 2: Configure SSIM Loss in YAML

Now that you've created the SSIM loss class, you need to configure it in your training YAML file. Here's how to set it up for RealESRGAN:

```yaml
# In your RealESRGAN training YAML
train:
  # Add SSIM loss alongside other losses
  ssim_opt:
    type: SSIMLoss
    loss_weight: 0.5  # Adjust weight as needed
    reduction: mean
    window_size: 11
    window_sigma: 1.5
    channel: 3
    is_rgb: true

  # Keep your existing losses
  pixel_opt:
    type: CharbonnierLoss  # Or L1Loss
    loss_weight: 1.0
    reduction: mean

  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv1_2': 0.1
      'conv2_2': 0.1
      'conv3_4': 1.0
      'conv4_4': 1.0
      'conv5_4': 1.0
    perceptual_weight: 1.0
    style_weight: 0.0
    criterion: l1

  gan_opt:
    type: GANLoss
    gan_type: hinge
    loss_weight: 0.1
```

## Step 3: Modify the Model to Use SSIM Loss

You need to update the model to use your new SSIM loss. There are two approaches:

### Option 1: Modify SRGANModel (Recommended)

Create a new file `basicsr/models/realesrgan_ssim_model.py`:



# Complete Guide to Adding SSIM Loss to RealESRGAN in BasicSR

I'll show you how to add SSIM loss to RealESRGAN in BasicSR with a complete implementation approach.

## Step 1: Create SSIM Loss Class

First, create a new file `basicsr/losses/ssim_loss.py` with this implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """SSIM loss, based on the SSIM metric implementation in basicsr.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        window_size (int): The size of the window for SSIM calculation. Default: 11.
        window_sigma (float): The sigma of Gaussian window. Default: 1.5.
        channel (int): Channel number of input tensors. Default: 3.
        is_rgb (bool): Whether the input is RGB image. Default: True.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 window_size=11,
                 window_sigma=1.5,
                 channel=3,
                 is_rgb=True):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        assert reduction in ['none', 'mean', 'sum']
        self.window_size = window_size
        self.window_sigma = window_sigma
        self.channel = channel
        self.is_rgb = is_rgb

        # Register buffer for window
        kernel = cv2.getGaussianKernel(window_size, window_sigma)
        window = np.outer(kernel, kernel.transpose())
        window = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window.expand(channel, 1, window_size, window_size))

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): Prediction with shape (N, C, H, W).
            target (Tensor): Ground truth with shape (N, C, H, W).
            weight (Tensor, optional): Element-wise weights. Default: None.

        Returns:
            Tensor: SSIM loss.
        """
        # SSIM returns similarity, so we use 1-SSIM as the loss
        ssim_value = self._ssim(pred, target)
        loss = 1.0 - ssim_value

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:  # 'none'
            return self.loss_weight * loss

    def _ssim(self, img1, img2):
        """Calculate SSIM for batched images.

        Args:
            img1 (Tensor): Images with range [0, 1], shape (N, C, H, W).
            img2 (Tensor): Images with range [0, 1], shape (N, C, H, W).

        Returns:
            Tensor: SSIM results with shape (N,).
        """
        # Scale to [0, 255] as in the original SSIM implementation
        img1 = img1 * 255.0
        img2 = img2 * 255.0

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        window = self.window
        window = window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

        if self.is_rgb:
            # Average over channels for RGB images
            ssim_per_channel = ssim_map.mean(dim=[2, 3])
            return ssim_per_channel.mean(dim=1)
        else:
            # For grayscale images
            return ssim_map.mean(dim=[1, 2, 3])
```

## Step 2: Create a Custom Model with SSIM Loss

Create a new model file `basicsr/models/realesrgan_ssim_model.py`:

```python
from collections import OrderedDict
import torch

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.losses import build_loss
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class RealESRGANSSIMModel(SRGANModel):
    """RealESRGAN model with additional SSIM loss."""

    def init_training_settings(self):
        # First initialize the original training settings from SRGANModel
        super(RealESRGANSSIMModel, self).init_training_settings()

        train_opt = self.opt['train']

        # Initialize SSIM loss if specified
        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

    def optimize_parameters(self, current_iter):
        # G
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # SSIM loss
            if self.cri_ssim:
                l_g_ssim = self.cri_ssim(self.output, self.gt)
                l_g_total += l_g_ssim
                loss_dict['l_g_ssim'] = l_g_ssim

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # D
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
```

## Step 3: Update [__init__.py](cci:7://file:///d:/personal%20projects/samsung/BasicSR/basicsr/__init__.py:0:0-0:0) Files

Make sure your new files are imported properly:

1. Add to [basicsr/losses/__init__.py](cci:7://file:///d:/personal%20projects/samsung/BasicSR/basicsr/losses/__init__.py:0:0-0:0):
   ```python
   from .ssim_loss import SSIMLoss
   ```

2. The model will be automatically imported by the registry system.

## Step 4: Create a Training YAML Configuration

Create a new YAML file, e.g., `options/train/RealESRGAN/train_realesrgan_x4_ssim.yml`:

```yaml
# RealESRGAN with SSIM loss
name: RealESRGAN_x4_SSIM
model_type: RealESRGANSSIMModel  # Use our custom model
scale: 4
num_gpu: 1  # Set to your GPU count

# Network definitions
network_g:
  type: RRDBNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  scale: 4

network_d:
  type: UNetDiscriminatorSN
  num_in_ch: 3
  num_feat: 64
  skip_connection: True

# Dataset settings
datasets:
  train:
    name: YourDataset
    type: PairedImageDataset
    dataroot_gt: datasets/your_dataset/train/hr
    dataroot_lq: datasets/your_dataset/train/lr
    gt_size: 256
    use_hflip: true
    use_rot: true
    io_backend:
      type: disk

# Training settings
train:
  # SSIM loss
  ssim_opt:
    type: SSIMLoss
    loss_weight: 0.5  # Adjust weight as needed
    reduction: mean
    window_size: 11
    window_sigma: 1.5
    channel: 3
    is_rgb: true

  # Charbonnier loss
  pixel_opt:
    type: CharbonnierLoss
    loss_weight: 1.0
    reduction: mean
    eps: 1e-12

  # Perceptual loss
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv1_2': 0.1
      'conv2_2': 0.1
      'conv3_4': 1.0
      'conv4_4': 1.0
      'conv5_4': 1.0
    perceptual_weight: 1.0
    style_weight: 0.0
    criterion: l1

  # GAN loss
  gan_opt:
    type: GANLoss
    gan_type: hinge  # RealESRGAN uses hinge loss
    loss_weight: 0.1

  # Optimizer settings
  optim_g:
    type: Adam
    lr: !!float 1e-4
    betas: [0.9, 0.99]
  optim_d:
    type: Adam
    lr: !!float 1e-4
    betas: [0.9, 0.99]

  # Training schedule
  net_d_iters: 1
  net_d_init_iters: 0
  ema_decay: 0.999  # EMA for generator

  # Training iterations
  scheduler:
    type: MultiStepLR
    milestones: [400000, 600000]
    gamma: 0.5
  total_iter: 800000

  # Validation settings
  val_freq: !!float 5e3
  save_checkpoint_freq: !!float 5e3

# Validation settings
val:
  val_freq: !!float 5e3
  save_img: true
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false

# Path settings
path:
  pretrain_network_g: ~  # Path to pretrained generator if available
  strict_load_g: true
  resume_state: ~  # Path to resume state if needed
```

## Step 5: Run Training

Start training with:

```bash
python basicsr/train.py -opt options/train/RealESRGAN/train_realesrgan_x4_ssim.yml
```

## Step 6: Inference

For inference, you don't need to modify anything. The trained model can be used with the standard inference scripts since the SSIM loss is only used during training:

```bash
python basicsr/test.py -opt options/test/RealESRGAN/test_realesrgan_x4.yml
```

## Key Points About SSIM Loss

1. **Why use SSIM loss?**
   - SSIM focuses on structural similarity rather than pixel-wise differences
   - Helps preserve textures and structures better than L1/L2 losses
   - Complements perceptual and adversarial losses for better visual quality

2. **Loss balancing:**
   - Start with `loss_weight: 0.5` for SSIM loss
   - Adjust weights based on validation results
   - SSIM loss typically works well with pixel losses (L1/Charbonnier)

3. **Training tips:**
   - SSIM loss may slow down convergence slightly
   - Consider pretraining with only pixel loss before adding SSIM
   - Monitor both PSNR and SSIM metrics during validation

This implementation gives you a complete solution for adding SSIM loss to RealESRGAN in the BasicSR framework, with all the necessary code and configuration files.