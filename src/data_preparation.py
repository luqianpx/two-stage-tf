import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.model_selection import KFold, train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADNIDataSplitter:
    """Class to handle ADNI data splitting operations"""
    def __init__(
        self,
        data_dir: str,
        label_file: str,
        test_size: float = 0.2,
        n_folds: int = 5,
        random_state: int = 42,
        valid_extensions: tuple = ('.nii.gz', '.nii')
    ):
        """
        Args:
            data_dir: Directory containing the MRI files
            label_file: Path to label CSV file
            test_size: Proportion of data to use for testing
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            valid_extensions: Tuple of valid file extensions
        """
        self.data_dir = Path(data_dir)
        self.label_file = Path(label_file)
        self.test_size = test_size
        self.n_folds = n_folds
        self.random_state = random_state
        self.valid_extensions = valid_extensions
        
        # Load labels
        self.labels_df = self._load_labels()
        
        # Initialize splits
        self.splits = {}
        self.subject_paths = []
        self.subject_labels = []

    def _load_labels(self) -> pd.DataFrame:
        """Load and validate label file"""
        try:
            df = pd.read_csv(self.label_file)
            required_columns = ['subject_id', 'label']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Label file must contain columns: {required_columns}")
            return df
        except Exception as e:
            logger.error(f"Error loading label file: {e}")
            raise

    def collect_data(self) -> None:
        """Collect all valid MRI files and their labels"""
        logger.info("Collecting data paths and labels...")
        
        # Get all MRI files
        mri_files = sorted(self.data_dir.glob('*.nii.gz'))
        if not mri_files:
            raise FileNotFoundError(f"No MRI files found in {self.data_dir}")

        # Match files with labels
        for file_path in tqdm(mri_files, desc="Loading dataset"):
            subject_id = file_path.stem.split('.')[0]
            if subject_id in self.labels_df['subject_id'].values:
                self.subject_paths.append(str(file_path))
                label = self.labels_df.loc[
                    self.labels_df['subject_id'] == subject_id, 'label'
                ].iloc[0]
                self.subject_labels.append(label)

        logger.info(f"Found {len(self.subject_paths)} valid subjects")

        # Convert to numpy arrays
        self.subject_paths = np.array(self.subject_paths)
        self.subject_labels = np.array(self.subject_labels)

    def create_splits(self) -> Dict:
        """Create train/test splits and k-fold splits"""
        if len(self.subject_paths) == 0:
            self.collect_data()

        logger.info("Creating data splits...")
        
        # Create train/test split
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.subject_paths,
            self.subject_labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.subject_labels
        )

        # Create k-fold splits
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_splits = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_paths)):
            fold_splits.append({
                'train': {
                    'paths': train_paths[train_idx],
                    'labels': train_labels[train_idx]
                },
                'val': {
                    'paths': train_paths[val_idx],
                    'labels': train_labels[val_idx]
                }
            })

        # Store all splits
        self.splits = {
            'test': {
                'paths': test_paths,
                'labels': test_labels
            },
            'folds': fold_splits,
            'num_classes': len(np.unique(self.subject_labels))
        }

        # Save split information
        self.save_split_info()
        
        return self.splits

    def save_split_info(self) -> None:
        """Save split information to CSV files"""
        output_dir = self.data_dir / "split_info"
        output_dir.mkdir(exist_ok=True)
        
        # Save test set info
        test_df = pd.DataFrame({
            'path': self.splits['test']['paths'],
            'label': self.splits['test']['labels']
        })
        test_df.to_csv(output_dir / "test_split.csv", index=False)
        
        # Save fold info
        for fold_idx, fold_data in enumerate(self.splits['folds']):
            fold_df = pd.DataFrame({
                'path': np.concatenate([
                    fold_data['train']['paths'],
                    fold_data['val']['paths']
                ]),
                'label': np.concatenate([
                    fold_data['train']['labels'],
                    fold_data['val']['labels']
                ]),
                'split': ['train'] * len(fold_data['train']['paths']) + 
                        ['val'] * len(fold_data['val']['paths'])
            })
            fold_df.to_csv(output_dir / f"fold_{fold_idx}.csv", index=False)

    def get_fold_data(self, fold_idx: int) -> Dict:
        """
        Get data for a specific fold
        
        Args:
            fold_idx: Index of the fold to retrieve
            
        Returns:
            Dictionary containing train and validation data for the fold
        """
        if not self.splits:
            self.create_splits()
            
        if fold_idx >= self.n_folds:
            raise ValueError(f"Fold index {fold_idx} is out of range (0-{self.n_folds-1})")
            
        return self.splits['folds'][fold_idx]

    def get_test_data(self) -> Dict:
        """Get test data"""
        if not self.splits:
            self.create_splits()
            
        return self.splits['test']

def prepare_adni_splits(
    data_dir: str,
    label_file: str,
    test_size: float = 0.2,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Convenience function to prepare ADNI data splits
    
    Args:
        data_dir: Directory containing the MRI files
        label_file: Path to label CSV file
        test_size: Proportion of data to use for testing
        n_folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing all split information
    """
    splitter = ADNIDataSplitter(
        data_dir=data_dir,
        label_file=label_file,
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state
    )
    
    return splitter.create_splits()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to MRI files directory")
    parser.add_argument("--label_file", type=str, required=True,
                       help="Path to label CSV file")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data for testing")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds for cross-validation")
    args = parser.parse_args()
    
    # Create splits
    splits = prepare_adni_splits(
        args.data_dir,
        args.label_file,
        args.test_size,
        args.n_folds
    )
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Number of classes: {splits['num_classes']}")
    print(f"Test set size: {len(splits['test']['paths'])}")
    print("\nFold information:")
    for i, fold in enumerate(splits['folds']):
        print(f"\nFold {i}:")
        print(f"Training samples: {len(fold['train']['paths'])}")
        print(f"Validation samples: {len(fold['val']['paths'])}")