# moco_pretrain.py
# This file is used to pretrain the MoCo model
# Author: px
# Date: 2021-04-01

import os
from argparse import ArgumentParser
from pathlib import Path
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from moco_model import MoCoModule
from resnet3d import resnet3d50
from data_module import MRIDataModule

def build_args(arg_defaults=None):
    """Build arguments for training"""
    pl.seed_everything(1234)
    
    # Default configurations
    data_config = Path.cwd() / "configs/data.yaml"
    arg_defaults = arg_defaults or {
        "accelerator": "gpu",
        "devices": -1,  # Use all available GPUs
        "strategy": "ddp",
        "max_epochs": 200,
        "num_workers": 8,
        "batch_size": 32,
        "precision": 16,  # Use mixed precision training
    }

    # ------------
    # Arguments
    # ------------
    parser = ArgumentParser()
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="your_dataset",
                       choices=["dataset1", "dataset2", "dataset3"])
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--im_size", default=128, type=int)
    
    # Add model specific arguments
    parser = MoCoModule.add_model_specific_args(parser)
    
    # Add trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)
    
    # Set defaults
    parser.set_defaults(**arg_defaults)
    
    # Parse arguments
    args = parser.parse_args()

    # Load data paths from config if not specified
    if args.dataset_dir is None:
        with open(data_config, "r") as f:
            paths = yaml.safe_load(f)["paths"]
        args.dataset_dir = paths.get(args.dataset_name)
        if args.dataset_dir is None:
            raise ValueError(f"Dataset path not found for {args.dataset_name}")

    return args

def main(args):
    """Main training function"""
    # Setup logging
    logger = setup_logging(args.log_dir, "moco_pretrain")
    
    # Create data module
    data_module = ADNIDataModule(
        data_dir=args.dataset_dir,
        label_file=args.label_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=(args.im_size, args.im_size, args.im_size),
        augment_train=True
    )
    
    # Initialize model
    model = MoCoModule(
        arch="resnet3d50",
        feature_dim=args.feature_dim,
        queue_size=args.queue_size,
        moco_momentum=args.moco_momentum,
        temperature=args.temperature,
        use_mlp=args.use_mlp,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.max_epochs
    )

    # Setup trainer
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=-1,
        strategy='ddp',
        precision=16,
        callbacks=[
            ModelCheckpoint(
                monitor='train_loss',
                dirpath=args.output_dir,
                filename='moco-{epoch:02d}-{train_loss:.2f}',
                save_top_k=3,
                mode='min',
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        logger=TensorBoardLogger(args.log_dir, name='moco')
    )

    # Train
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # Get arguments
    args = build_args()
    
    # Run training
    main(args)