# data_module.py
# This file is used to create the data module for the ADNI dataset
# Author: px
# Date: 2021-04-01

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict
from pathlib import Path

from dataset import ADNIDataset
from data_preparation import ADNIDataSplitter
from data_augmentation import MRI3DAugmentation

class ADNIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        label_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        target_shape: tuple = (128, 128, 128),
        test_size: float = 0.2,
        n_folds: int = 5,
        fold_idx: Optional[int] = None,
        augment_train: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.label_file = Path(label_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_shape = target_shape
        self.test_size = test_size
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.augment_train = augment_train
        self.seed = seed
        
        # Initialize data splitter
        self.splitter = ADNIDataSplitter(
            data_dir=self.data_dir,
            label_file=self.label_file,
            test_size=self.test_size,
            n_folds=self.n_folds,
            random_state=self.seed
        )
        
        # Initialize augmentation
        if self.augment_train:
            self.transform = MRI3DAugmentation(
                rotation_range=15,
                flip_prob=0.5,
                scale_range=(0.9, 1.1),
                translate_range=(-0.1, 0.1),
                noise_std=0.02
            )
        else:
            self.transform = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        if stage == 'fit' or stage is None:
            # Get fold data if fold_idx is specified
            if self.fold_idx is not None:
                fold_data = self.splitter.get_fold_data(self.fold_idx)
                
                # Create training dataset
                self.train_dataset = ADNIDataset(
                    data_dir=self.data_dir,
                    label_file=self.label_file,
                    target_shape=self.target_shape
                )
                self.train_dataset.subjects = [Path(p) for p in fold_data['train']['paths']]
                self.train_dataset.labels = fold_data['train']['labels']
                
                # Create validation dataset
                self.val_dataset = ADNIDataset(
                    data_dir=self.data_dir,
                    label_file=self.label_file,
                    target_shape=self.target_shape
                )
                self.val_dataset.subjects = [Path(p) for p in fold_data['val']['paths']]
                self.val_dataset.labels = fold_data['val']['labels']
            
            else:
                # Use all data for pretraining
                self.train_dataset = ADNIDataset(
                    data_dir=self.data_dir,
                    label_file=self.label_file,
                    target_shape=self.target_shape
                )
                self.val_dataset = None

        if stage == 'test' or stage is None:
            # Get test data
            test_data = self.splitter.get_test_data()
            
            self.test_dataset = ADNIDataset(
                data_dir=self.data_dir,
                label_file=self.label_file,
                target_shape=self.target_shape
            )
            self.test_dataset.subjects = [Path(p) for p in test_data['paths']]
            self.test_dataset.labels = test_data['labels']

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )