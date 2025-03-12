"""
Utility functions for the project
author: px
date: 2021-03-09
version: 1.0
"""

import torch
import random
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List

def setup_logging(log_dir: str, name: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth.tar'
):
    """Save checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'model_best.pth.tar'
        torch.save(state, best_filepath)