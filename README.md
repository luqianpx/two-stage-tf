# ADNI MRI Classification Project

## Overview
This repository provides a complete pipeline for self-supervised learning and classification on 3D MRI data, specifically using the ADNI (Alzheimer’s Disease Neuroimaging Initiative) dataset. It includes data preparation, augmentation, model pretraining using MoCo (Momentum Contrast), fine-tuning, evaluation, and interpretability using 3D Grad-CAM.

## Author px
## Please connect with me if you have any questions or suggestions. I'd hppay to discuss this project with you. This project is still under development and I will update the code and documentation as I go along.
## Project Structure

## Features
### Configuration and Utilities
- **`config.py`** - Configuration management using YAML files, supporting inheritance and overrides.
- **`utils.py`** - Utility functions for logging, seed setting, and model checkpoint saving.

### Data Processing Modules
- **`data_augmentation.py`** - Augments 3D MRI data to generate positive and negative pairs for contrastive learning.
- **`data_preparation.py`** - Splits the ADNI dataset into training, validation, and test sets, supporting k-fold cross-validation.
- **`dataset.py`** - Custom PyTorch dataset for loading and preprocessing 3D MRI volumes.
- **`data_module.py`** - PyTorch Lightning data module to facilitate data loading, augmentation, and splitting.

### Model Implementation and Training
- **`moco_model.py`** - Adaptation of MoCo (Momentum Contrast) for self-supervised learning on 3D medical images.
- **`moco_pretrain.py`** - Pretraining script for self-supervised learning using Momentum Contrast.
- **`fine_tune.py`** - Fine-tunes a MoCo-pretrained encoder for classification using labeled ADNI data.
- **`resnet3d.py`** - Implementation of 3D ResNet for medical imaging, integrating SE attention and dropout.

### Model Evaluation and Interpretability
- **`evaluate.py`** - Evaluates trained models and generates Grad-CAM visualizations.
- **`grad_3d.py`** - Implements Grad-CAM for 3D models to provide visual explanations.

### Testing and Integration
- **`test_integration.py`** - End-to-end testing of the entire pipeline using pytest.

## Project Structure
.src
├── config.py # Configuration management
├── data_augmentation.py # 3D data augmentation
├── data_preparation.py # Data splitting and preparation
├── dataset.py # ADNI dataset loading
├── data_module.py # data module for the ADNI dataset
├── evaluate.py # Model evaluation
├── test_integration.py # end-to-end test for the entire ADNI project pipeline using pytest
├── fine_tune.py # Fine-tuning script
├── grad_3d.py # Grad-CAM visualization
├── moco_model.py # MoCo model implementation
├── moco_pretrain.py # MoCo pretraining script
├── resnet3d.py # 3D ResNet architecture
└── utils.py # Shared utilities

## Usage 
1. Pretraining with MoCo:

python moco_pretrain.py \
--config configs/pretrain_config.yaml

2. Fine-tuning:

python fine_tune.py \
--config configs/finetune_config.yaml
--moco_checkpoint path/to/moco_checkpoint.ckpt

3. Evaluation:

python evaluate.py \
--model_path path/to/finetuned_model.ckpt \
--test_data path/to/test_set.csv \
--output_dir evaluation_results
