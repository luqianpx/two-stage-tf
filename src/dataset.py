# This is the dataset class for the ADNI dataset
# author:px
# date:2022-01-07
# version:1.0

import os
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adni_niitonumpy(nifti_file, target_shape=(128, 128, 128)):
    """
    Convert NIfTI file to preprocessed NumPy array for ResNet3D.
    
    Args:
        nifti_file (str or Path): Path to NIfTI file
        target_shape (tuple): Target shape for the MRI volume (D, H, W)
    
    Returns:
        np.ndarray: Preprocessed MRI data with shape (D, H, W)
    """
    try:
        # Load NIfTI file
        img = nib.load(str(nifti_file))
        img_data = img.get_fdata()
        
        # Normalize to [0, 1]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        
        # Resize if necessary using scipy's zoom
        if img_data.shape != target_shape:
            factors = [t / s for t, s in zip(target_shape, img_data.shape)]
            img_data = zoom(img_data, factors, order=3)  # order=3 for cubic interpolation
            
        return img_data.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error processing NIfTI file {nifti_file}: {e}")
        raise

class ADNIDataset(Dataset):
    """ADNI Dataset for 3D MRI data compatible with ResNet3D"""
    
    def __init__(self, data_dir, label_file, target_shape=(128, 128, 128)):
        """
        Args:
            data_dir (str): Directory containing MRI files
            label_file (str): Path to label CSV file
            target_shape (tuple): Target shape for the MRI volumes (D, H, W)
        """
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        
        # Load labels
        try:
            self.labels_df = pd.read_csv(label_file)
            required_columns = ['subject_id', 'label']
            if not all(col in self.labels_df.columns for col in required_columns):
                raise ValueError(f"Label file must contain columns: {required_columns}")
        except Exception as e:
            logger.error(f"Error loading label file: {e}")
            raise
            
        # Get all MRI files
        self.file_list = sorted(self.data_dir.glob('*.nii.gz'))  # Adjust pattern if needed
        if not self.file_list:
            raise FileNotFoundError(f"No MRI files found in {data_dir}")
        
        # Match files with labels
        self.subjects = []
        self.labels = []
        
        for file_path in tqdm(self.file_list, desc="Loading dataset"):
            subject_id = file_path.stem.split('.')[0]
            if subject_id in self.labels_df['subject_id'].values:
                self.subjects.append(file_path)
                label = self.labels_df.loc[
                    self.labels_df['subject_id'] == subject_id, 'label'
                ].iloc[0]
                self.labels.append(label)
        
        logger.info(f"Loaded {len(self.subjects)} valid subjects")
    
    def __len__(self):
        return len(self.subjects)
        
    def __getitem__(self, idx):
        try:
            # Load and preprocess MRI data
            img_path = self.subjects[idx]
            img = adni_niitonumpy(img_path, self.target_shape)
            
            # Add channel dimension for ResNet3D (channels_last format)
            img = np.expand_dims(img, axis=0)  # Shape: (1, D, H, W)
            img = torch.from_numpy(img).float()  # Convert to tensor
        
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return img, label
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {img_path}: {e}")
            raise

def parallel_load_mri_3D(data_dir, label_file, batch_size=16, num_workers=4):
    """
    Load 3D MRI data with parallel processing for ResNet3D.
    
    Args:
        data_dir (str): Directory containing MRI files
        label_file (str): Path to label CSV file
        batch_size (int): Batch size for loading
        num_workers (int): Number of parallel workers
    """
    try:
        # Create dataset
        dataset = ADNIDataset(
            data_dir=data_dir,
            label_file=label_file
        )
        
        # Configure dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        logger.info(f"""DataLoader configured with:
            Batch size: {batch_size}
            Number of workers: {num_workers}
            Number of batches: {len(dataloader)}""")
        
        return dataset, dataloader
        
    except Exception as e:
        logger.error(f"Error setting up data loading: {e}")
        raise

if __name__ == "__main__":
    DATA_DIR = "path/to/mri/files"
    LABEL_FILE = "path/to/labels.csv"
    
    try:
        dataset, dataloader = parallel_load_mri_3D(
            data_dir=DATA_DIR,
            label_file=LABEL_FILE,
            batch_size=16,
            num_workers=4
        )
        
        # Test loading a few batches
        for i, (images, labels) in enumerate(dataloader):
            logger.info(f"Batch {i}: Image shape: {images.shape}, Labels: {labels.shape}")
            if i >= 2:
                break
                
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
