"""
PyroCast Fire Spread Model - Step 6: Evaluate Model
====================================================
Comprehensive evaluation of the trained fire spread model:
1. Quantitative metrics (IoU, Dice, temporal accuracy)
2. Per-day performance analysis
3. Visualization of predictions vs ground truth
4. Error analysis and failure case identification

Output: Evaluation report with metrics and visualizations
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# Import dataset creation function
import importlib.util
spec = importlib.util.spec_from_file_location("build_dataset", os.path.join(os.path.dirname(__file__), "04_build_dataset.py"))
build_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset_module)
create_tf_dataset = build_dataset_module.create_tf_dataset

import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Output directory for evaluation results
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


class ModelEvaluator:
    """
    Comprehensive model evaluation for fire spread predictions.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to trained model. If None, uses latest model.
        """
        if model_path is None:
            # Find latest model
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.ptkeras')]
            if not model_files:
                raise FileNotFoundError("No trained model found in MODEL_DIR")
            model_path = os.path.join(MODEL_DIR, sorted(model_files)[-1])
        
        logger.info(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'temporal_weighted_loss': self._dummy_loss,
                'IoUMetric': self._dummy_metric,
                'TemporalAccuracy': self._dummy_metric
            }
        )
        self.model_path = model_path
        
        # Metrics storage
        self.metrics = {}
        
    def _dummy_loss(self, y_true, y_pred):
        """Placeholder for loading models with custom loss."""
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def _dummy_metric(self, y_true, y_pred):
        """Placeholder for loading models with custom metrics."""
        return 0.0
    
    def compute_iou(self, y_true, y_pred, threshold=0.5):
        """Compute Intersection over Union."""
        y_pred_binary = (y_pred > threshold).astype(np.float32)
        y_true_binary = (y_true > 0.5).astype(np.float32)
        
        intersection = np.sum(y_pred_binary * y_true_binary)
        union = np.sum(y_pred_binary) + np.sum(y_true_binary) - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def compute_dice(self, y_true, y_pred, threshold=0.5):
        """Compute Dice coefficient."""
        y_pred_binary = (y_pred > threshold).astype(np.float32)
        y_true_binary = (y_true > 0.5).astype(np.float32)
        
        intersection = np.sum(y_pred_binary * y_true_binary)
        total = np.sum(y_pred_binary) + np.sum(y_true_binary)
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2 * intersection / total
    
    def compute_precision_recall(self, y_true, y_pred, threshold=0.5):
        """Compute precision and recall."""
        y_pred_binary = (y_pred > threshold).astype(np.float32)
        y_true_binary = (y_true > 0.5).astype(np.float32)
        
        true_positives = np.sum(y_pred_binary * y_true_binary)
        predicted_positives = np.sum(y_pred_binary)
        actual_positives = np.sum(y_true_binary)
        
        precision = true_positives / (predicted_positives + 1e-6)
        recall = true_positives / (actual_positives + 1e-6)
        
        return precision, recall
    
    def evaluate_dataset(self, dataset, name="test"):
        """
        Evaluate model on a dataset and compute comprehensive metrics.
        
        Args:
            dataset: tf.data.Dataset
            name: Name for this evaluation (e.g., 'test', 'val')
        
        Returns:
            dict with all computed metrics
        """
        logger.info(f"Evaluating on {name} set...")
        
        # Collect predictions and ground truth
        all_y_true = []
        all_y_pred = []
        
        for batch_x, batch_y in dataset:
            predictions = self.model.predict(batch_x, verbose=0)
            all_y_true.append(batch_y.numpy())
            all_y_pred.append(predictions)
        
        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        
        logger.info(f"Evaluated {len(y_true)} samples")
        
        # Compute overall metrics
        metrics = {
            'name': name,
            'num_samples': len(y_true),
            'overall': {},
            'per_day': []
        }
        
        # Overall metrics (all days combined)
        metrics['overall']['iou'] = float(self.compute_iou(y_true, y_pred))
        metrics['overall']['dice'] = float(self.compute_dice(y_true, y_pred))
        
        precision, recall = self.compute_precision_recall(y_true, y_pred)
        metrics['overall']['precision'] = float(precision)
        metrics['overall']['recall'] = float(recall)
        metrics['overall']['f1'] = float(2 * precision * recall / (precision + recall + 1e-6))
        
        # Per-day metrics
        for day in range(PREDICTION_DAYS):
            y_true_day = y_true[:, day, :, :, :]
            y_pred_day = y_pred[:, day, :, :, :]
            
            day_metrics = {
                'day': day + 1,
                'iou': float(self.compute_iou(y_true_day, y_pred_day)),
                'dice': float(self.compute_dice(y_true_day, y_pred_day)),
            }
            
            precision, recall = self.compute_precision_recall(y_true_day, y_pred_day)
            day_metrics['precision'] = float(precision)
            day_metrics['recall'] = float(recall)
            day_metrics['f1'] = float(2 * precision * recall / (precision + recall + 1e-6))
            
            # Compute IoU at different thresholds
            for thresh in IOU_THRESHOLDS:
                day_metrics[f'iou_{int(thresh*100)}'] = float(
                    self.compute_iou(y_true_day, y_pred_day, threshold=thresh)
                )
            
            metrics['per_day'].append(day_metrics)
        
        self.metrics[name] = metrics
        
        # Store predictions for visualization
        self._y_true = y_true
        self._y_pred = y_pred
        
        return metrics
    
    def visualize_predictions(self, num_samples=5, output_dir=None):
        """
        Generate visualizations of predictions vs ground truth.
        
        Creates:
        1. Side-by-side comparison grids
        2. Temporal progression animations
        3. Error heatmaps
        """
        if output_dir is None:
            output_dir = EVAL_DIR
        
        if not hasattr(self, '_y_true') or not hasattr(self, '_y_pred'):
            logger.warning("No predictions available. Run evaluate_dataset first.")
            return
        
        y_true = self._y_true
        y_pred = self._y_pred
        
        # Select random samples
        indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)
        
        for idx in indices:
            self._visualize_sample(y_true[idx], y_pred[idx], idx, output_dir)
        
        # Generate summary plots
        self._plot_metrics_summary(output_dir)
        self._plot_per_day_metrics(output_dir)
        
        logger.info(f"Visualizations saved to: {output_dir}")
    
    def _visualize_sample(self, y_true, y_pred, sample_idx, output_dir):
        """Visualize a single sample's predictions."""
        
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, PREDICTION_DAYS + 1, figure=fig, wspace=0.05, hspace=0.15)
        
        # Create fire colormap
        fire_cmap = mcolors.LinearSegmentedColormap.from_list(
            'fire', ['#000000', '#8B0000', '#FF4500', '#FFD700', '#FFFFFF']
        )
        
        # Row 1: Ground Truth
        fig.add_subplot(gs[0, 0]).set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        fig.add_subplot(gs[0, 0]).set_title('Day →', fontsize=10)
        
        for day in range(PREDICTION_DAYS):
            ax = fig.add_subplot(gs[0, day + 1])
            im = ax.imshow(y_true[day, :, :, 0], cmap=fire_cmap, vmin=0, vmax=1)
            ax.set_title(f'Day {day + 1}', fontsize=10)
            ax.axis('off')
        
        # Row 2: Predictions
        fig.add_subplot(gs[1, 0]).set_ylabel('Prediction', fontsize=12, fontweight='bold')
        
        for day in range(PREDICTION_DAYS):
            ax = fig.add_subplot(gs[1, day + 1])
            im = ax.imshow(y_pred[day, :, :, 0], cmap=fire_cmap, vmin=0, vmax=1)
            
            # Compute IoU for this day
            iou = self.compute_iou(y_true[day], y_pred[day])
            ax.set_xlabel(f'IoU: {iou:.3f}', fontsize=9)
            ax.axis('off')
        
        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Fire Probability')
        
        plt.suptitle(f'Fire Spread Prediction - Sample {sample_idx}', fontsize=14, fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, f'prediction_sample_{sample_idx}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create error visualization
        self._visualize_errors(y_true, y_pred, sample_idx, output_dir)
    
    def _visualize_errors(self, y_true, y_pred, sample_idx, output_dir):
        """Visualize prediction errors."""
        
        fig, axes = plt.subplots(2, PREDICTION_DAYS, figsize=(3 * PREDICTION_DAYS, 6))
        
        for day in range(PREDICTION_DAYS):
            gt = y_true[day, :, :, 0]
            pred = y_pred[day, :, :, 0]
            
            # Binary masks
            gt_binary = gt > 0.5
            pred_binary = pred > 0.5
            
            # Error types
            # Green: True Positive, Red: False Positive, Blue: False Negative
            error_map = np.zeros((*gt.shape, 3))
            error_map[gt_binary & pred_binary] = [0, 1, 0]  # TP - Green
            error_map[~gt_binary & pred_binary] = [1, 0, 0]  # FP - Red
            error_map[gt_binary & ~pred_binary] = [0, 0, 1]  # FN - Blue
            
            # Probability error
            prob_error = pred - gt
            
            axes[0, day].imshow(error_map)
            axes[0, day].set_title(f'Day {day + 1} Errors')
            axes[0, day].axis('off')
            
            im = axes[1, day].imshow(prob_error, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, day].set_title(f'Day {day + 1} Prob Error')
            axes[1, day].axis('off')
        
        # Legend for error types
        fig.text(0.02, 0.5, 'Green=TP, Red=FP, Blue=FN', fontsize=8, rotation=90, va='center')
        
        plt.colorbar(im, ax=axes[1, :], shrink=0.8, label='Probability Error')
        plt.suptitle(f'Error Analysis - Sample {sample_idx}', fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, f'error_sample_{sample_idx}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_summary(self, output_dir):
        """Plot summary of overall metrics."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of overall metrics
        for name, metrics in self.metrics.items():
            overall = metrics['overall']
            x = list(overall.keys())
            y = list(overall.values())
            
            axes[0].bar(x, y, alpha=0.7, label=name)
        
        axes[0].set_ylabel('Score')
        axes[0].set_title('Overall Metrics')
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        
        # IoU at different thresholds
        for name, metrics in self.metrics.items():
            if 'per_day' in metrics:
                # Average IoU at different thresholds across days
                for thresh in IOU_THRESHOLDS:
                    key = f'iou_{int(thresh*100)}'
                    ious = [d.get(key, 0) for d in metrics['per_day']]
                    axes[1].plot(range(1, PREDICTION_DAYS + 1), ious, 
                               marker='o', label=f'{name} (τ={thresh})')
        
        axes[1].set_xlabel('Day')
        axes[1].set_ylabel('IoU')
        axes[1].set_title('IoU at Different Thresholds')
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=150)
        plt.close()
    
    def _plot_per_day_metrics(self, output_dir):
        """Plot metrics breakdown by prediction day."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = ['iou', 'dice', 'precision', 'recall']
        
        for ax, metric_name in zip(axes.flat, metrics_to_plot):
            for name, metrics in self.metrics.items():
                if 'per_day' in metrics:
                    days = [d['day'] for d in metrics['per_day']]
                    values = [d[metric_name] for d in metrics['per_day']]
                    ax.plot(days, values, marker='o', linewidth=2, label=name)
            
            ax.set_xlabel('Prediction Day')
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f'{metric_name.upper()} by Day')
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_day_metrics.png'), dpi=150)
        plt.close()
    
    def save_report(self, output_dir=None):
        """Save comprehensive evaluation report."""
        
        if output_dir is None:
            output_dir = EVAL_DIR
        
        report = {
            'model_path': self.model_path,
            'evaluation_time': datetime.now().isoformat(),
            'config': {
                'img_size': IMG_SIZE,
                'prediction_days': PREDICTION_DAYS,
                'input_channels': INPUT_CHANNELS,
                'iou_thresholds': IOU_THRESHOLDS
            },
            'results': self.metrics
        }
        
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Also create text summary
        self._create_text_summary(output_dir)
        
        return report
    
    def _create_text_summary(self, output_dir):
        """Create human-readable text summary."""
        
        lines = [
            "=" * 60,
            "PYROCAST FIRE SPREAD MODEL - EVALUATION REPORT",
            "=" * 60,
            f"\nModel: {os.path.basename(self.model_path)}",
            f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n" + "=" * 60,
            "CONFIGURATION",
            "=" * 60,
            f"Image Size: {IMG_SIZE}x{IMG_SIZE}",
            f"Prediction Days: {PREDICTION_DAYS}",
            f"Input Channels: {INPUT_CHANNELS}",
        ]
        
        for name, metrics in self.metrics.items():
            lines.extend([
                "\n" + "=" * 60,
                f"RESULTS - {name.upper()}",
                "=" * 60,
                f"\nNumber of Samples: {metrics['num_samples']}",
                "\nOverall Metrics:",
            ])
            
            for metric, value in metrics['overall'].items():
                lines.append(f"  {metric}: {value:.4f}")
            
            lines.append("\nPer-Day Metrics:")
            lines.append("-" * 50)
            lines.append(f"{'Day':<6} {'IoU':<10} {'Dice':<10} {'Prec':<10} {'Recall':<10}")
            lines.append("-" * 50)
            
            for day_metrics in metrics['per_day']:
                lines.append(
                    f"{day_metrics['day']:<6} "
                    f"{day_metrics['iou']:<10.4f} "
                    f"{day_metrics['dice']:<10.4f} "
                    f"{day_metrics['precision']:<10.4f} "
                    f"{day_metrics['recall']:<10.4f}"
                )
        
        lines.extend([
            "\n" + "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])
        
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))
        
        # Also print to console
        print('\n'.join(lines))


def main():
    """Main evaluation entry point."""
    
    logger.info("=" * 60)
    logger.info("PyroCast Fire Spread Model - Evaluation")
    logger.info("=" * 60)
    
    # Find latest model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    
    if not model_files:
        logger.error("No trained model found. Run 05_train_model.py first.")
        return
    
    latest_model = os.path.join(MODEL_DIR, sorted(model_files)[-1])
    logger.info(f"Using model: {latest_model}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(latest_model)
    
    # Load test dataset
    test_path = os.path.join(TFRECORD_DIR, "spread_test.tfrecord")
    
    if os.path.exists(test_path):
        test_ds = create_tf_dataset(test_path, BATCH_SIZE, shuffle=False)
        evaluator.evaluate_dataset(test_ds, name='test')
    else:
        logger.warning(f"Test dataset not found at {test_path}")
    
    # Load validation dataset
    val_path = os.path.join(TFRECORD_DIR, "spread_val.tfrecord")
    
    if os.path.exists(val_path):
        val_ds = create_tf_dataset(val_path, BATCH_SIZE, shuffle=False)
        evaluator.evaluate_dataset(val_ds, name='validation')
    
    # Generate visualizations
    evaluator.visualize_predictions(num_samples=10)
    
    # Save report
    evaluator.save_report()
    
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {EVAL_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
