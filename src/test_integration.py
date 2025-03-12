import pytest
import torch
from pathlib import Path

from dataset import ADNIDataset
from data_preparation import ADNIDataSplitter
from moco_model import MoCoModule
from fine_tune import FineTuneModel
from evaluate import ModelEvaluator

def test_full_pipeline():
    """Test the entire pipeline from data loading to evaluation"""
    # 1. Load configuration
    config = load_test_config()
    
    # 2. Test data loading
    dataset = ADNIDataset(
        data_dir=config['data']['data_dir'],
        label_file=config['data']['label_file']
    )
    assert len(dataset) > 0
    
    # 3. Test data splitting
    splitter = ADNIDataSplitter(
        data_dir=config['data']['data_dir'],
        label_file=config['data']['label_file']
    )
    splits = splitter.create_splits()
    assert len(splits['folds']) == config['data']['n_folds']
    
    # 4. Test MoCo pretraining
    moco_model = MoCoModule(config['moco'])
    assert isinstance(moco_model, MoCoModule)
    
    # 5. Test fine-tuning
    fine_tune_model = FineTuneModel(
        moco_checkpoint_path=config['model']['pretrained_path'],
        num_classes=config['data']['num_classes']
    )
    assert isinstance(fine_tune_model, FineTuneModel)
    
    # 6. Test evaluation
    evaluator = ModelEvaluator(
        model_path=config['evaluation']['model_path'],
        test_data_path=config['evaluation']['test_data']
    )
    assert isinstance(evaluator, ModelEvaluator)