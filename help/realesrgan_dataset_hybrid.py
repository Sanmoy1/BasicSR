import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANHybridDataset(data.Dataset):
    """Hybrid Dataset for Real-ESRGAN that supports both paired and unpaired data sources.
    
    This dataset can handle:
    1. Unpaired data (GT only) with synthetic degradation - like RealESRGANDataset
    2. Paired data (LQ-GT pairs) - like RealESRGANPairedDataset
    
    The dataset will randomly sample from both sources based on specified probabilities.
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            # Common settings
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            gt_size (int): Cropped patched size for gt patches.
            scale (int): Scale factor.
            
            # Unpaired data settings (for synthetic degradation)
            dataroot_gt_unpaired (str): Data root path for unpaired gt images.
            meta_info_unpaired (str): Path for meta information file for unpaired data.
            unpaired_prob (float): Probability of sampling from unpaired data (0.0-1.0).
            
            # Paired data settings
            dataroot_paired (str): Base data root path for paired images (contains both GT and LQ).
            meta_info_paired (str): Path for meta information file for paired data (format: "gt_path, lq_path" per line).
            filename_tmpl (str): Template for each filename for paired data.
            
            # Degradation settings for unpaired data (same as RealESRGANDataset)
            blur_kernel_size, kernel_list, kernel_prob, blur_sigma, etc.
    """

    def __init__(self, opt):
        super(RealESRGANHybridDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        # Probability of using unpaired vs paired data
        self.unpaired_prob = opt.get('unpaired_prob', 0.5)
        
        # Initialize unpaired data paths (GT only, for synthetic degradation)
        self.unpaired_paths = []
        if 'dataroot_gt_unpaired' in opt and opt['dataroot_gt_unpaired']:
            self.gt_folder_unpaired = opt['dataroot_gt_unpaired']
            
            if self.io_backend_opt['type'] == 'lmdb':
                # LMDB backend for unpaired data
                if not self.gt_folder_unpaired.endswith('.lmdb'):
                    raise ValueError(f"'dataroot_gt_unpaired' should end with '.lmdb', but received {self.gt_folder_unpaired}")
                with open(osp.join(self.gt_folder_unpaired, 'meta_info.txt')) as fin:
                    self.unpaired_paths = [line.split('.')[0] for line in fin]
            else:
                # Disk backend for unpaired data
                if 'meta_info_unpaired' in opt and opt['meta_info_unpaired']:
                    with open(opt['meta_info_unpaired']) as fin:
                        paths = [line.strip().split(' ')[0] for line in fin]
                        self.unpaired_paths = [os.path.join(self.gt_folder_unpaired, v) for v in paths]
        
        # Initialize paired data paths (LQ-GT pairs)
        self.paired_paths = []
        if 'dataroot_paired' in opt and opt['dataroot_paired']:
            self.paired_folder = opt['dataroot_paired']
            self.filename_tmpl = opt.get('filename_tmpl', '{}')
            
            if self.io_backend_opt['type'] == 'lmdb':
                # LMDB backend for paired data (not commonly used with single folder approach)
                raise NotImplementedError("LMDB backend not supported with single paired folder approach. Use separate GT/LQ folders or disk backend.")
            elif 'meta_info_paired' in opt and opt['meta_info_paired']:
                # Meta info file for paired data (format: "gt_path, lq_path" per line)
                with open(opt['meta_info_paired']) as fin:
                    paths = [line.strip() for line in fin]
                self.paired_paths = []
                for path in paths:
                    gt_path, lq_path = path.split(', ')
                    gt_path = os.path.join(self.paired_folder, gt_path)
                    lq_path = os.path.join(self.paired_folder, lq_path)
                    self.paired_paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
            else:
                # Without meta info, assume GT and LQ are in separate subfolders
                gt_subfolder = os.path.join(self.paired_folder, 'GT')
                lq_subfolder = os.path.join(self.paired_folder, 'LQ')
                if os.path.exists(gt_subfolder) and os.path.exists(lq_subfolder):
                    self.paired_paths = paired_paths_from_folder([lq_subfolder, gt_subfolder], ['lq', 'gt'], self.filename_tmpl)
                else:
                    raise ValueError(f"When using single paired folder without meta_info_paired, GT and LQ subfolders must exist at {gt_subfolder} and {lq_subfolder}")
        
        # Total dataset size
        self.total_unpaired = len(self.unpaired_paths)
        self.total_paired = len(self.paired_paths)
        
        if self.total_unpaired == 0 and self.total_paired == 0:
            raise ValueError("No valid data found. Please check your data paths.")
        
        # Degradation settings for unpaired data (same as RealESRGANDataset)
        if self.total_unpaired > 0:
            # First degradation
            self.blur_kernel_size = opt['blur_kernel_size']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']
            self.blur_sigma = opt['blur_sigma']
            self.betag_range = opt['betag_range']
            self.betap_range = opt['betap_range']
            self.sinc_prob = opt['sinc_prob']
            
            # Second degradation
            self.blur_kernel_size2 = opt['blur_kernel_size2']
            self.kernel_list2 = opt['kernel_list2']
            self.kernel_prob2 = opt['kernel_prob2']
            self.blur_sigma2 = opt['blur_sigma2']
            self.betag_range2 = opt['betag_range2']
            self.betap_range2 = opt['betap_range2']
            self.sinc_prob2 = opt['sinc_prob2']
            
            # Final sinc filter
            self.final_sinc_prob = opt['final_sinc_prob']
            
            self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
            self.pulse_tensor = torch.zeros(21, 21).float()
            self.pulse_tensor[10, 10] = 1

    def _load_unpaired_data(self, index):
        """Load unpaired data and generate synthetic degradation kernels."""
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # Load GT image
        gt_path = self.unpaired_paths[index % self.total_unpaired]
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                index = random.randint(0, self.total_unpaired - 1)
                gt_path = self.unpaired_paths[index]
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        
        img_gt = imfrombytes(img_bytes, float32=True)
        
        # Augmentation
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        
        # Crop or pad to specified size
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.opt.get('gt_size', 400)
        
        # Pad if needed
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        
        # Crop if needed
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
        
        # Generate degradation kernels (same as RealESRGANDataset)
        # First degradation kernel
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list, self.kernel_prob, kernel_size,
                self.blur_sigma, self.blur_sigma, [-math.pi, math.pi],
                self.betag_range, self.betap_range, noise_range=None)
        
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # Second degradation kernel
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2, self.kernel_prob2, kernel_size,
                self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2, self.betap_range2, noise_range=None)
        
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # Final sinc kernel
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
        
        # Convert to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        
        return {
            'gt': img_gt,
            'kernel1': kernel,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path,
            'data_type': 'unpaired'
        }

    def _load_paired_data(self, index):
        """Load paired LQ-GT data."""
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        scale = self.opt['scale']
        
        # Load GT and LQ images
        path_info = self.paired_paths[index % self.total_paired]
        gt_path = path_info['gt_path']
        lq_path = path_info['lq_path']
        
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        
        # Augmentation for training
        if self.opt.get('phase', 'train') == 'train':
            gt_size = self.opt.get('gt_size', 256)
            # Random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # Flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        
        # Convert to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'data_type': 'paired'
        }

    def __getitem__(self, index):
        """Get data item. Randomly choose between paired and unpaired data."""
        # Decide whether to use paired or unpaired data
        if self.total_unpaired == 0:
            # Only paired data available
            return self._load_paired_data(index)
        elif self.total_paired == 0:
            # Only unpaired data available
            return self._load_unpaired_data(index)
        else:
            # Both types available, choose randomly
            if random.random() < self.unpaired_prob:
                return self._load_unpaired_data(index)
            else:
                return self._load_paired_data(index)

    def __len__(self):
        """Return the maximum length of both datasets."""
        return max(self.total_unpaired, self.total_paired, 1)


# Example configuration for using the hybrid dataset with two-path approach:
"""
datasets:
  train:
    name: Hybrid_DF2K_DIV2K
    type: RealESRGANHybridDataset
    
    # ============= UNPAIRED DATA SETTINGS (for synthetic degradation) =============
    # Path to high-quality images for synthetic degradation
    dataroot_gt_unpaired: datasets/DF2K
    meta_info_unpaired: datasets/DF2K/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt
    unpaired_prob: 0.7  # 70% chance to use unpaired data
    
    # ============= PAIRED DATA SETTINGS (single base path approach) =============
    # Base path containing both GT and LQ images
    dataroot_paired: datasets/DIV2K_paired
    # Meta info file with format: "gt_relative_path, lq_relative_path" per line
    # Example line: "HR/image001.png, LR_bicubic/X2/image001x2.png"
    meta_info_paired: datasets/DIV2K_paired/meta_info_paired.txt
    filename_tmpl: '{}'
    
    # Alternative: Without meta_info_paired, use GT/ and LQ/ subfolders:
    # dataroot_paired: datasets/DIV2K_paired  # Should contain GT/ and LQ/ subfolders
    # meta_info_paired: ~  # Leave empty to use subfolder approach
    
    # ============= COMMON SETTINGS =============
    io_backend:
      type: disk
    gt_size: 256
    use_hflip: True
    use_rot: False
    scale: 2
    
    # ============= DEGRADATION SETTINGS (for unpaired data only) =============
    blur_kernel_size: 21
    kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob: 0.1
    blur_sigma: [0.2, 3]
    betag_range: [0.5, 4]
    betap_range: [1, 2]
    
    blur_kernel_size2: 21
    kernel_list2: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob2: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob2: 0.1
    blur_sigma2: [0.2, 1.5]
    betag_range2: [0.5, 4]
    betap_range2: [1, 2]
    
    final_sinc_prob: 0.8
    
    # Data loader
    num_worker_per_gpu: 5
    batch_size_per_gpu: 12
    dataset_enlarge_ratio: 1
    prefetch_mode: ~
"""