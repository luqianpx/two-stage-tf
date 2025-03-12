"""
Configuration management for the project
author: px
date: 2021-03-09
version: 1.0
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy

class Config:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration with inheritance"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle inheritance
        if 'inherit_from' in config:
            parent_path = config.pop('inherit_from')
            parent_config = Config.load_config(parent_path)
            # Deep merge parent and child configs
            config = Config._deep_merge(parent_config, config)
            
        return config
    
    @staticmethod
    def _deep_merge(parent: Dict, child: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = deepcopy(parent)
        for key, value in child.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

# Usage example:
if __name__ == "__main__":
    # Load different configs
    pretrain_config = Config.load_config('configs/pretrain_config.yaml')
    finetune_config = Config.load_config('configs/finetune_config.yaml')
    evaluate_config = Config.load_config('configs/evaluate_config.yaml')
    
    # Access config values
    print(f"MoCo feature dimension: {pretrain_config['moco']['feature_dim']}")
    print(f"Fine-tuning learning rate: {finetune_config['training']['learning_rate']}")
    print(f"Number of GradCAM samples: {evaluate_config['visualization']['gradcam']['num_samples']}")