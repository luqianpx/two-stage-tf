# Extract the trained encoder from MoCo
# author:px
# date:2022-01-07
# version:1.0
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, Optional

from moco_model import MoCoModule

from dataset import ADNIDataset
from resnet3d import resnet3d50
from data_module import ADNIDataModule
from data_preparation import ADNIDataSplitter

# This file is used to fine-tune the MoCo model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuneModel(pl.LightningModule):
    """Fine-tuning model using pretrained MoCo backbone"""
    
    def __init__(
        self,
        moco_checkpoint_path: str,
        num_classes: int = 2,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        feature_dim: int = 512,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained MoCo model
        self.backbone = self._load_moco_backbone(moco_checkpoint_path)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _load_moco_backbone(self, checkpoint_path: str) -> nn.Module:
        """Load MoCo backbone from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load MoCo model
        moco = MoCoModule.load_from_checkpoint(checkpoint_path)
        
        # Get encoder_q without projection head
        backbone = moco.model.encoder_q
        if isinstance(backbone.fc, nn.Sequential):
            backbone.fc = nn.Identity()  # Remove MLP head
        else:
            backbone.fc = nn.Identity()  # Remove linear classifier
            
        return backbone

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        metrics = {
            'val/loss': loss,
            'val/acc': acc
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        return metrics

    def configure_optimizers(self):
        # Optimizer
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }

def train_fold(args, fold_idx: int, run_dir: Path) -> Dict:
    """Train model on a single fold"""
    logger.info(f"Training Fold {fold_idx + 1}/{args.n_folds}")
    
    # Create data module for this fold
    data_module = ADNIDataModule(
        data_dir=args.dataset_dir,
        label_file=args.label_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=(args.im_size, args.im_size, args.im_size),
        n_folds=args.n_folds,
        fold_idx=fold_idx,
        augment_train=True
    )
    
    # Setup data
    data_module.setup('fit')
    
    # Create model
    model = FineTuneModel(
        moco_checkpoint_path=args.moco_checkpoint,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone
    )
    
    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=str(run_dir / f"fold_{fold_idx}"),
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=-1,
        precision=16,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=str(run_dir / f"fold_{fold_idx}"),
                filename='model-{epoch:02d}-{val_loss:.2f}',
                save_top_k=1,
                mode='min',
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        logger=TensorBoardLogger(
            save_dir=run_dir,
            name=f"fold_{fold_idx}"
        )
    )
    
    # Train
    trainer.fit(model, data_module)
    
    return {
        'fold': fold_idx,
        'best_val_acc': trainer.callback_metrics['val_acc'].item(),
        'model_path': trainer.checkpoint_callback.best_model_path
    }

def main():
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--moco_checkpoint", type=str, required=True,
                       help="Path to MoCo checkpoint")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--label_file", type=str, required=True,
                       help="Path to label file")
    parser.add_argument("--im_size", type=int, default=128,
                       help="Input image size")
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                       help="Weight decay")
    parser.add_argument("--freeze_backbone", type=bool, default=True,
                       help="Freeze backbone weights")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of epochs")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds for cross-validation")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data for testing")
    
    # Add trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/finetune_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create data splitter
    splitter = ADNIDataSplitter(
        data_dir=args.dataset_dir,
        label_file=args.label_file,
        test_size=args.test_size,
        n_folds=args.n_folds
    )
    
    # Create splits
    splits = splitter.create_splits()

    # Train each fold
    fold_results = []
    for fold_idx in range(args.n_folds):
        # Get fold data
        fold_data = splitter.get_fold_data(fold_idx)
        
        # Train fold
        fold_result = train_fold(args, fold_data, fold_idx, run_dir)
        fold_results.append(fold_result)

    # Save fold results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(run_dir / "fold_results.csv", index=False)
    
    # Get best fold
    best_fold = results_df.loc[results_df['best_val_acc'].idxmax()]
    logger.info(f"Best fold: {best_fold['fold']} with accuracy: {best_fold['best_val_acc']:.4f}")
    
    # Save test set for future evaluation
    test_data = splitter.get_test_data()
    test_df = pd.DataFrame({
        'path': test_data['paths'],
        'label': test_data['labels']
    })
    test_df.to_csv(run_dir / "test_set.csv", index=False)

if __name__ == "__main__":
    main()