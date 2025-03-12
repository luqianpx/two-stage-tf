# evaluate.py
# This file is used to evaluate the model on the test set
# It also generates Grad-CAM visualizations for the test set
# It uses the fine_tune.py model and the data_module.py data module
# Author: px
# Date: 2021-04-01

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
import nibabel as nib

from fine_tune import FineTuneModel
from data_module import ADNIDataModule
from grad_3d import GradCAM3D
from utils import setup_logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class to handle model evaluation and visualization"""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        label_file: str,
        output_dir: str,
        config: dict,
        device: str = "cuda"
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing MRI files
            label_file: Path to label CSV file
            output_dir: Directory to save evaluation results
            config: Configuration dictionary
            device: Device to run evaluation on
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Setup logging
        self.logger = setup_logging(
            self.output_dir / "logs",
            "evaluation"
        )
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup data
        self.data_module = ADNIDataModule(
            data_dir=data_dir,
            label_file=label_file,
            batch_size=config['evaluation']['batch_size'],
            num_workers=config['data']['num_workers'],
            target_shape=tuple(config['data']['target_shape']),
            augment_train=False
        )
        self.data_module.setup('test')
        
        # Initialize Grad-CAM
        self.gradcam = GradCAM3D(
            self.model,
            self.model.backbone.layer4  # Assuming ResNet backbone
        )
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {model_path}")
        try:
            model = FineTuneModel.load_from_checkpoint(model_path)
            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def evaluate(self) -> dict:
        """Run evaluation"""
        logger.info("Starting evaluation...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data_module.test_dataloader(), desc="Evaluating"):
                images, labels = batch
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability of positive class
                
        # Calculate metrics
        metrics = self._calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        # Save results
        self._save_metrics(metrics)
        self._generate_visualizations(metrics)
        
        return metrics
        
    def _calculate_metrics(self, labels, predictions, probabilities) -> dict:
        """Calculate evaluation metrics"""
        # Basic metrics
        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        auc = roc_auc_score(labels, probabilities)
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp)
        sensitivity = recall
        ppv = precision
        npv = tn / (tn + fn)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'ppv': ppv,
            'npv': npv,
            'f1_score': f1,
            'auc_score': auc,
            'confusion_matrix': conf_matrix
        }
        
    def _save_metrics(self, metrics: dict):
        """Save metrics to file"""
        # Save numerical metrics
        metrics_df = pd.DataFrame([{
            k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
        }])
        metrics_df.to_csv(self.output_dir / "metrics.csv", index=False)
        
        # Save confusion matrix
        np.save(self.output_dir / "confusion_matrix.npy", metrics['confusion_matrix'])
        
    def _generate_visualizations(self, metrics: dict):
        """Generate and save visualizations"""
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title('Confusion Matrix')
        plt.savefig(vis_dir / "confusion_matrix.png")
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {metrics["auc_score"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(vis_dir / "roc_curve.png")
        plt.close()
        
    def generate_gradcam_visualizations(self, num_samples: int = 5):
        """Generate Grad-CAM visualizations"""
        logger.info("Generating Grad-CAM visualizations...")
        
        vis_dir = self.output_dir / "gradcam"
        vis_dir.mkdir(exist_ok=True)
        
        # Create summary file
        summary_file = vis_dir / "visualization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Grad-CAM Visualization Summary\n")
            f.write("============================\n\n")
        
        # Get samples
        test_loader = self.data_module.test_dataloader()
        for i, (data, label) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data = data.to(self.device)
            
            # Generate heatmap
            heatmap = self.gradcam.generate(data)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1).item()
                conf = probs[0][pred].item()
            
            # Save visualizations
            self._save_gradcam_sample(
                data[0],
                heatmap[0],
                label.item(),
                pred,
                conf,
                i,
                vis_dir,
                summary_file
            )
            
    def _save_gradcam_sample(
        self,
        original: torch.Tensor,
        heatmap: torch.Tensor,
        true_label: int,
        pred_label: int,
        confidence: float,
        sample_idx: int,
        vis_dir: Path,
        summary_file: Path
    ):
        """Save Grad-CAM sample visualizations"""
        # Convert to numpy
        original = original.cpu().numpy()
        heatmap = heatmap.cpu().numpy()
        
        # Save as NIfTI
        affine = np.eye(4)
        nib.save(
            nib.Nifti1Image(original, affine),
            str(vis_dir / f"sample_{sample_idx}_original.nii.gz")
        )
        nib.save(
            nib.Nifti1Image(heatmap, affine),
            str(vis_dir / f"sample_{sample_idx}_heatmap.nii.gz")
        )
        
        # Create overlay
        overlay = original * 0.7 + heatmap * 0.3
        nib.save(
            nib.Nifti1Image(overlay, affine),
            str(vis_dir / f"sample_{sample_idx}_overlay.nii.gz")
        )
        
        # Update summary file
        with open(summary_file, 'a') as f:
            f.write(f"\nSample {sample_idx}:\n")
            f.write(f"True Label: {true_label}\n")
            f.write(f"Predicted Label: {pred_label}\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            f.write("----------------------------------------\n")

def main():
    """Main function"""
    import argparse
    from config import Config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save evaluation results")
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load_config(args.config)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=config['paths']['data_dir'],
        label_file=config['paths']['label_file'],
        output_dir=args.output_dir,
        config=config
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Generate Grad-CAM visualizations
    if config['visualization']['gradcam']['enabled']:
        evaluator.generate_gradcam_visualizations(
            config['visualization']['gradcam']['num_samples']
        )
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()