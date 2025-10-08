# Knowledge Distillation for Denoising Models

## Overview

This guide explains how to implement knowledge distillation to transfer knowledge from a larger, better-performing teacher model to a lightweight student model for image denoising. This approach helps you create a compact model that maintains high performance while being suitable for deployment.

### Key Concepts

- **Teacher Model**: A larger, pre-trained model with superior denoising performance
- **Student Model**: Your lightweight architecture that will learn from the teacher
- **Distillation Loss**: Combines task loss (pixel-level) with knowledge transfer loss (feature matching)

## 1. Architecture Setup

### Step 1.1: Create Distillation Model Class

Create a new file `basicsr/models/distillation_model.py`:

```python
import torch
from collections import OrderedDict
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DistillationModel(SRModel):
    """Knowledge Distillation Model for Image Denoising/Super-Resolution.
    
    This model trains a lightweight student network using knowledge from a 
    pre-trained teacher network.
    """

    def __init__(self, opt):
        super(DistillationModel, self).__init__(opt)
        
        # Build teacher network
        if self.is_train:
            self.net_teacher = build_network(opt['network_teacher'])
            self.net_teacher = self.model_to_device(self.net_teacher)
            self.print_network(self.net_teacher)
            
            # Load pre-trained teacher weights
            teacher_path = self.opt['path'].get('pretrain_network_teacher')
            if teacher_path is not None:
                self.load_network(self.net_teacher, teacher_path, 
                                self.opt['path'].get('strict_load_teacher', True), 
                                param_key='params')
            
            # Set teacher to eval mode and freeze parameters
            self.net_teacher.eval()
            for param in self.net_teacher.parameters():
                param.requires_grad = False

    def init_training_settings(self):
        """Initialize training settings including distillation losses."""
        super(DistillationModel, self).init_training_settings()
        
        train_opt = self.opt['train']
        
        # Initialize distillation loss (feature matching)
        if train_opt.get('distillation_opt'):
            distill_type = train_opt['distillation_opt']['type']
            
            if distill_type == 'FeatureDistillation':
                self.cri_distill = FeatureDistillationLoss(
                    loss_weight=train_opt['distillation_opt'].get('loss_weight', 1.0),
                    feature_layers=train_opt['distillation_opt'].get('feature_layers', None),
                    criterion=train_opt['distillation_opt'].get('criterion', 'l2')
                ).to(self.device)
            elif distill_type == 'OutputDistillation':
                self.cri_distill = build_loss(train_opt['distillation_opt']).to(self.device)
            else:
                raise NotImplementedError(f'Distillation type {distill_type} not implemented')
        else:
            self.cri_distill = None

    def optimize_parameters(self, current_iter):
        """Optimize student network with both task loss and distillation loss."""
        self.optimizer_g.zero_grad()
        
        # Forward pass - student
        self.output = self.net_g(self.lq)
        
        l_total = 0
        loss_dict = OrderedDict()
        
        # Task loss (pixel loss)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        # Perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        # Distillation loss
        if self.cri_distill:
            with torch.no_grad():
                teacher_output = self.net_teacher(self.lq)
            
            l_distill = self.cri_distill(self.output, teacher_output)
            l_total += l_distill
            loss_dict['l_distill'] = l_distill
        
        l_total.backward()
        self.optimizer_g.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        """Test using only the student network."""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def save(self, epoch, current_iter):
        """Save only the student network."""
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
        if self.ema_decay > 0:
            self.save_network(self.net_g_ema, 'net_g_ema', current_iter)


class FeatureDistillationLoss(torch.nn.Module):
    """Feature-based distillation loss.
    
    Matches intermediate features between teacher and student networks.
    """
    
    def __init__(self, loss_weight=1.0, feature_layers=None, criterion='l2'):
        super(FeatureDistillationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.feature_layers = feature_layers
        self.criterion = criterion
        
        if criterion == 'l1':
            self.loss_fn = torch.nn.L1Loss()
        elif criterion == 'l2':
            self.loss_fn = torch.nn.MSELoss()
        elif criterion == 'smooth_l1':
            self.loss_fn = torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f'Unsupported criterion: {criterion}')
    
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: Output from student network
            teacher_output: Output from teacher network (detached)
        """
        loss = self.loss_fn(student_output, teacher_output.detach())
        return self.loss_weight * loss
```

### Step 1.2: Create Advanced Feature Distillation (Optional)

For more sophisticated distillation with intermediate features:

```python
class AdvancedFeatureDistillation(torch.nn.Module):
    """Advanced feature distillation with attention transfer.
    
    This requires modifying your networks to expose intermediate features.
    """
    
    def __init__(self, loss_weight=1.0, attention_weight=0.5):
        super(AdvancedFeatureDistillation, self).__init__()
        self.loss_weight = loss_weight
        self.attention_weight = attention_weight
        self.mse_loss = torch.nn.MSELoss()
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: Dict of intermediate features from student
            teacher_features: Dict of intermediate features from teacher
        """
        total_loss = 0
        
        # Feature matching loss
        for key in student_features.keys():
            if key in teacher_features:
                s_feat = student_features[key]
                t_feat = teacher_features[key].detach()
                
                # Spatial attention transfer
                s_attention = self.compute_attention(s_feat)
                t_attention = self.compute_attention(t_feat)
                
                # Feature loss
                feat_loss = self.mse_loss(s_feat, t_feat)
                # Attention loss
                attn_loss = self.mse_loss(s_attention, t_attention)
                
                total_loss += feat_loss + self.attention_weight * attn_loss
        
        return self.loss_weight * total_loss
    
    def compute_attention(self, features):
        """Compute spatial attention maps."""
        # Sum across channels and normalize
        attention = torch.sum(features ** 2, dim=1, keepdim=True)
        attention = F.normalize(attention.view(attention.size(0), -1), p=2, dim=1)
        attention = attention.view(attention.size(0), 1, features.size(2), features.size(3))
        return attention
```

## 2. Configuration Setup

### Step 2.1: Create Training YAML Configuration

Create `options/train/Distillation/train_distillation_denoising.yml`:

```yaml
# General settings
name: Distillation_Denoising_Student
model_type: DistillationModel
scale: 1  # For denoising, scale is 1
num_gpu: 1
manual_seed: 0

# Dataset settings
datasets:
  train:
    name: TrainDataset
    type: RealESRGANHybridDataset  # Use your hybrid dataset
    dataroot_gt_unpaired: datasets/your_unpaired_gt
    dataroot_gt_paired: datasets/your_paired_gt
    dataroot_lq_paired: datasets/your_paired_lq
    meta_info_unpaired: datasets/meta_info_unpaired.txt
    meta_info_paired: datasets/meta_info_paired.txt
    
    io_backend:
      type: disk
    
    # Data augmentation
    use_hflip: true
    use_rot: true
    gt_size: 400  # Consistent crop size
    
    # Degradation settings (for unpaired data)
    blur_kernel_size: 21
    kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob: 0.1
    blur_sigma: [0.2, 3]
    betag_range: [0.5, 4]
    betap_range: [1, 2]
    
    # First degradation
    resize_prob: [0.2, 0.7, 0.1]
    resize_range: [0.15, 1.5]
    gaussian_noise_prob: 0.5
    noise_range: [1, 30]
    poisson_scale_range: [0.05, 3]
    gray_noise_prob: 0.4
    jpeg_range: [30, 95]
    
    # Second degradation
    second_blur_prob: 0.8
    resize_prob2: [0.3, 0.4, 0.3]
    resize_range2: [0.3, 1.2]
    gaussian_noise_prob2: 0.5
    noise_range2: [1, 25]
    poisson_scale_range2: [0.05, 2.5]
    gray_noise_prob2: 0.4
    jpeg_range2: [30, 95]
    
    # Dataloader settings
    num_worker_per_gpu: 4
    batch_size_per_gpu: 4
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: ValDataset
    type: PairedImageDataset
    dataroot_gt: datasets/val/gt
    dataroot_lq: datasets/val/lq
    io_backend:
      type: disk

# Network definitions
network_g:
  # Student network (your lightweight model)
  type: YourLightweightArchitecture  # Replace with your architecture name
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 32  # Smaller than teacher
  num_block: 8   # Fewer blocks than teacher
  # Add your specific architecture parameters

network_teacher:
  # Teacher network (larger, pre-trained model)
  type: RRDBNet  # Or your larger architecture
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  scale: 1

# Path settings
path:
  pretrain_network_g: ~  # Optional: path to pretrained student
  pretrain_network_teacher: experiments/pretrained_models/teacher_model.pth  # REQUIRED: Teacher weights
  strict_load_g: true
  strict_load_teacher: true
  resume_state: ~

# Training settings
train:
  ema_decay: 0.999
  
  # Optimizer for student
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  
  # Learning rate scheduler
  scheduler:
    type: MultiStepLR
    milestones: [50000, 100000, 150000, 200000]
    gamma: 0.5
  
  total_iter: 250000
  warmup_iter: -1  # No warmup
  
  # Pixel loss (task loss)
  pixel_opt:
    type: CharbonnierLoss
    loss_weight: 1.0
    reduction: mean
    eps: !!float 1e-12
  
  # Perceptual loss (optional)
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv1_2': 0.1
      'conv2_2': 0.1
      'conv3_4': 1.0
      'conv4_4': 1.0
      'conv5_4': 1.0
    vgg_type: vgg19
    use_input_norm: true
    perceptual_weight: 1.0
    style_weight: 0
    range_norm: false
    criterion: l1
  
  # Distillation loss (knowledge transfer)
  distillation_opt:
    type: OutputDistillation  # or FeatureDistillation
    loss_weight: 2.0  # Weight for distillation loss (tune this)
    criterion: l2  # l1, l2, or smooth_l1
    # For FeatureDistillation:
    # feature_layers: ['layer1', 'layer2', 'layer3']  # Specify which layers to match

# Validation settings
val:
  val_freq: !!float 5e3
  save_img: true
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false

# Logging settings
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# Distributed training settings
dist_params:
  backend: nccl
  port: 29500
```

## 3. Training Steps

### Step 3.1: Prepare Your Models

1. **Teacher Model**: Ensure you have a trained teacher model saved as `.pth` file
   ```bash
   # Your teacher model should be at:
   experiments/pretrained_models/teacher_model.pth
   ```

2. **Student Model**: Your lightweight architecture should be registered in BasicSR
   ```python
   # In basicsr/archs/your_student_arch.py
   from basicsr.utils.registry import ARCH_REGISTRY
   
   @ARCH_REGISTRY.register()
   class YourLightweightArchitecture(nn.Module):
       def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=32, num_block=8):
           super(YourLightweightArchitecture, self).__init__()
           # Your architecture implementation
   ```

### Step 3.2: Register the Distillation Model

Add to `basicsr/models/__init__.py`:

```python
from .distillation_model import DistillationModel

__all__ = [
    # ... existing models
    'DistillationModel',
]
```

### Step 3.3: Start Training

```bash
# Single GPU training
python basicsr/train.py -opt options/train/Distillation/train_distillation_denoising.yml

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
basicsr/train.py -opt options/train/Distillation/train_distillation_denoising.yml --launcher pytorch
```

### Step 3.4: Monitor Training

Check training logs and tensorboard:

```bash
tensorboard --logdir experiments/Distillation_Denoising_Student/tb_logger
```

Key metrics to monitor:
- `l_pix`: Task loss (should decrease)
- `l_distill`: Distillation loss (should decrease)
- `psnr`: Validation PSNR (should increase)
- `ssim`: Validation SSIM (should increase)

## 4. Advanced Techniques

### 4.1: Temperature Scaling (for Soft Targets)

If you want to use soft targets with temperature scaling:

```python
class SoftTargetDistillation(torch.nn.Module):
    """Distillation with temperature scaling for soft targets."""
    
    def __init__(self, loss_weight=1.0, temperature=3.0):
        super(SoftTargetDistillation, self).__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_output, teacher_output):
        # Apply temperature scaling
        student_soft = F.log_softmax(student_output / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_output / self.temperature, dim=1)
        
        # KL divergence loss
        loss = self.kl_div(student_soft, teacher_soft.detach()) * (self.temperature ** 2)
        return self.loss_weight * loss
```

### 4.2: Progressive Distillation

Start with higher distillation weight and gradually reduce:

```yaml
train:
  distillation_opt:
    type: OutputDistillation
    loss_weight: 5.0  # Start high
    
  # Add scheduler for distillation weight
  distillation_scheduler:
    milestones: [50000, 100000, 150000]
    gamma: 0.5  # Reduce by half at each milestone
```

### 4.3: Hint-based Distillation

For matching intermediate features with different dimensions:

```python
class HintDistillation(torch.nn.Module):
    """Hint-based distillation with 1x1 conv for dimension matching."""
    
    def __init__(self, student_channels, teacher_channels, loss_weight=1.0):
        super(HintDistillation, self).__init__()
        self.loss_weight = loss_weight
        
        # 1x1 conv to match dimensions
        self.regressor = torch.nn.Conv2d(student_channels, teacher_channels, 1)
        self.mse_loss = torch.nn.MSELoss()
    
    def forward(self, student_feature, teacher_feature):
        # Match dimensions
        student_aligned = self.regressor(student_feature)
        
        # Compute loss
        loss = self.mse_loss(student_aligned, teacher_feature.detach())
        return self.loss_weight * loss
```

## 5. Evaluation and Deployment

### Step 5.1: Test the Student Model

```bash
python basicsr/test.py -opt options/test/test_student_model.yml
```

Test configuration (`options/test/test_student_model.yml`):

```yaml
name: Test_Student_Denoising
model_type: SRModel  # Use base SRModel for testing
scale: 1
num_gpu: 1

datasets:
  test:
    name: TestDataset
    type: PairedImageDataset
    dataroot_gt: datasets/test/gt
    dataroot_lq: datasets/test/lq
    io_backend:
      type: disk

network_g:
  type: YourLightweightArchitecture
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 32
  num_block: 8

path:
  pretrain_network_g: experiments/Distillation_Denoising_Student/models/net_g_250000.pth
  strict_load_g: true

val:
  save_img: true
  suffix: ~
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false
```

### Step 5.2: Compare Performance

Create a comparison script:

```python
# compare_models.py
import torch
from basicsr.archs import build_network

# Load teacher
teacher_opt = {'type': 'RRDBNet', 'num_feat': 64, 'num_block': 23, ...}
teacher = build_network(teacher_opt)
teacher.load_state_dict(torch.load('teacher.pth')['params'])

# Load student
student_opt = {'type': 'YourLightweightArchitecture', 'num_feat': 32, ...}
student = build_network(student_opt)
student.load_state_dict(torch.load('student.pth')['params'])

# Compare parameters
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")
print(f"Compression ratio: {teacher_params / student_params:.2f}x")

# Compare inference time
import time
dummy_input = torch.randn(1, 3, 256, 256).cuda()

teacher.eval().cuda()
student.eval().cuda()

# Warmup
for _ in range(10):
    _ = teacher(dummy_input)
    _ = student(dummy_input)

# Measure teacher
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = teacher(dummy_input)
torch.cuda.synchronize()
teacher_time = (time.time() - start) / 100

# Measure student
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = student(dummy_input)
torch.cuda.synchronize()
student_time = (time.time() - start) / 100

print(f"Teacher inference time: {teacher_time*1000:.2f} ms")
print(f"Student inference time: {student_time*1000:.2f} ms")
print(f"Speedup: {teacher_time / student_time:.2f}x")
```

## 6. Tips and Best Practices

### Hyperparameter Tuning

1. **Distillation Weight**: Start with `loss_weight: 2.0` and adjust based on validation performance
   - Too high: Student overfits to teacher, poor generalization
   - Too low: Student doesn't learn from teacher effectively

2. **Temperature**: For soft targets, use `temperature: 3.0-5.0`
   - Higher temperature: Softer probability distribution
   - Lower temperature: Closer to hard targets

3. **Learning Rate**: Use slightly higher LR for student than typical training
   - Student: `2e-4`
   - Normal training: `1e-4`

### Common Issues

1. **Student not learning**: Increase distillation weight or check teacher model quality
2. **Student overfitting to teacher**: Reduce distillation weight, add more task loss
3. **Slow convergence**: Increase learning rate or use warmup
4. **Memory issues**: Reduce batch size or use gradient checkpointing

### Recommended Training Schedule

```
Phase 1 (0-50k iters): High distillation weight (5.0)
Phase 2 (50k-100k iters): Medium distillation weight (2.0)
Phase 3 (100k-200k iters): Low distillation weight (1.0)
Phase 4 (200k+ iters): Fine-tune with task loss only (0.5)
```

## 7. Summary

This distillation pipeline allows you to:

1. ✅ Train a lightweight student model using a pre-trained teacher
2. ✅ Maintain high performance while reducing model size
3. ✅ Support both output-level and feature-level distillation
4. ✅ Work with your existing hybrid dataset (paired + unpaired)
5. ✅ Integrate seamlessly with BasicSR framework

The key files you need:
- `basicsr/models/distillation_model.py` - Distillation model implementation
- `options/train/Distillation/train_distillation_denoising.yml` - Training configuration
- Pre-trained teacher model weights (`.pth` file)

Start training and monitor the balance between task loss and distillation loss to achieve optimal student performance!