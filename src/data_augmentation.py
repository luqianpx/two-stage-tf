# this is the data augmentation class
# author:px
# date:2022-01-07
# version:1.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import elasticdeform
import scipy.ndimage as ndimage
from torchvision.transforms import RandomApply
import random

class MRI3DAugmentation:
    """
    3D MRI Data Augmentation class for self-supervised learning
    Supports generation of positive pairs through different augmentations
    and tracking of augmentation identity for contrastive learning
    """
    
    def __init__(self,
                 rotation_range: int = 15,
                 flip_prob: float = 0.5,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 translate_range: Tuple[float, float] = (-0.1, 0.1),
                 noise_std: float = 0.02,
                 gamma_range: Tuple[float, float] = (0.8, 1.2),
                 elastic_alpha_range: Tuple[float, float] = (100, 200),
                 elastic_sigma_range: Tuple[float, float] = (9, 13),
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 p_augment: float = 0.5):
        """
        Args:
            rotation_range: Maximum rotation angle in degrees
            flip_prob: Probability of random flipping
            scale_range: Range for random scaling
            translate_range: Range for random translation as fraction of image size
            noise_std: Standard deviation for Gaussian noise
            gamma_range: Range for random gamma correction
            elastic_alpha_range: Range for elastic deformation magnitude
            elastic_sigma_range: Range for elastic deformation smoothness
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            p_augment: Probability of applying each augmentation
        """
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.noise_std = noise_std
        self.gamma_range = gamma_range
        self.elastic_alpha_range = elastic_alpha_range
        self.elastic_sigma_range = elastic_sigma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p_augment = p_augment
        
        # Track augmentation identity for contrastive learning
        self.current_sample_id = None
    
    def _random_rotation_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random 3D rotation"""
        angles = [
            np.random.uniform(-self.rotation_range, self.rotation_range) 
            for _ in range(3)
        ]
        volume = ndimage.rotate(volume, angles[0], axes=(1, 2), reshape=False)
        volume = ndimage.rotate(volume, angles[1], axes=(0, 2), reshape=False)
        volume = ndimage.rotate(volume, angles[2], axes=(0, 1), reshape=False)
        return torch.from_numpy(volume.astype(np.float32))

    def _random_flip_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random 3D flipping"""
        axes = [0, 1, 2]
        for axis in axes:
            if random.random() < self.flip_prob:
                volume = torch.flip(volume, [axis])
        return volume

    def _random_scale_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random 3D scaling"""
        scale_factor = np.random.uniform(*self.scale_range)
        return F.interpolate(
            volume.unsqueeze(0),
            scale_factor=scale_factor,
            mode='trilinear',
            align_corners=True
        ).squeeze(0)

    def _random_translate_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random 3D translation"""
        shift = [
            np.random.uniform(*self.translate_range) * s 
            for s in volume.shape
        ]
        return torch.from_numpy(
            ndimage.shift(volume.numpy(), shift, mode='reflect')
        )

    def _add_gaussian_noise(self, volume: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise"""
        noise = torch.randn_like(volume) * self.noise_std
        return volume + noise

    def _random_gamma(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random gamma correction"""
        gamma = np.random.uniform(*self.gamma_range)
        mn = volume.min()
        range = volume.max() - mn
        volume = ((volume - mn) / range) ** gamma * range + mn
        return volume

    def _elastic_transform(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation"""
        alpha = np.random.uniform(*self.elastic_alpha_range)
        sigma = np.random.uniform(*self.elastic_sigma_range)
        volume = elasticdeform.deform_random_grid(
            volume.numpy(),
            sigma=sigma,
            points=3,
            axis=(0, 1, 2),
            order=2,
            mode='reflect'
        )
        return torch.from_numpy(volume.astype(np.float32))

    def _adjust_brightness_contrast(self, volume: torch.Tensor) -> torch.Tensor:
        """Adjust brightness and contrast"""
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        mean = volume.mean()
        volume = (volume - mean) * contrast + mean * brightness
        return volume

    def generate_pair(self, volume: torch.Tensor, sample_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a positive pair from the same volume using different augmentations
        
        Args:
            volume: Input 3D MRI volume
            sample_id: Unique identifier for the input volume
            
        Returns:
            Tuple of (augmented_volume1, augmented_volume2)
        """
        self.current_sample_id = sample_id
        
        # List of augmentation functions
        augmentations = [
            self._random_rotation_3d,
            self._random_flip_3d,
            self._random_scale_3d,
            self._random_translate_3d,
            self._add_gaussian_noise,
            self._random_gamma,
            self._elastic_transform,
            self._adjust_brightness_contrast
        ]
        
        # Apply random augmentations to create two views
        aug_volume1 = volume.clone()
        aug_volume2 = volume.clone()
        
        for aug_fn in augmentations:
            if random.random() < self.p_augment:
                aug_volume1 = aug_fn(aug_volume1)
            if random.random() < self.p_augment:
                aug_volume2 = aug_fn(aug_volume2)
        
        return aug_volume1, aug_volume2

    def check_pair_relationship(self, id1: int, id2: int) -> int:
        """
        Check if two augmented volumes are from the same original volume
        
        Args:
            id1: Sample ID of first volume
            id2: Sample ID of second volume
            
        Returns:
            1 if positive pair (same original volume)
            0 if negative pair (different original volumes)
        """
        return 1 if id1 == id2 else 0

class ContrastiveMRIDataset(torch.utils.data.Dataset):
    """Dataset class for contrastive learning with 3D MRI data"""
    
    def __init__(self, 
                 volumes: List[torch.Tensor],
                 augmentation: MRI3DAugmentation):
        """
        Args:
            volumes: List of 3D MRI volumes
            augmentation: MRI3DAugmentation instance
        """
        self.volumes = volumes
        self.augmentation = augmentation
        
    def __len__(self) -> int:
        return len(self.volumes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        volume = self.volumes[idx]
        
        # Generate positive pair
        aug1, aug2 = self.augmentation.generate_pair(volume, idx)
        
        # For negative pair, get a different random volume
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.volumes) - 1)
        neg_volume = self.volumes[neg_idx]
        neg_aug, _ = self.augmentation.generate_pair(neg_volume, neg_idx)
        
        return {
            'anchor': aug1,
            'positive': aug2,
            'negative': neg_aug,
            'pos_label': torch.tensor(1, dtype=torch.float32),
            'neg_label': torch.tensor(0, dtype=torch.float32)
        }

# test useage of the data augmentation class
if __name__ == "__main__":
    # Create sample data
    sample_volume = torch.randn(128, 128, 128)
    volumes = [torch.randn(128, 128, 128) for _ in range(10)]
    
    # Initialize augmentation
    augmentation = MRI3DAugmentation(
        rotation_range=15,
        flip_prob=0.5,
        scale_range=(0.9, 1.1),
        translate_range=(-0.1, 0.1),
        noise_std=0.02,
        gamma_range=(0.8, 1.2),
        elastic_alpha_range=(100, 200),
        elastic_sigma_range=(9, 13),
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        p_augment=0.5
    )
    
    # Create dataset
    dataset = ContrastiveMRIDataset(volumes, augmentation)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    # Test the dataloader
    for batch in dataloader:
        print("Anchor shape:", batch['anchor'].shape)
        print("Positive shape:", batch['positive'].shape)
        print("Negative shape:", batch['negative'].shape)
        print("Positive label:", batch['pos_label'])
        print("Negative label:", batch['neg_label'])
        break
