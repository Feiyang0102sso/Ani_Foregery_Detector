import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from .abstract_class import AbstractEvaluator

class ImageConfusionMatrix(AbstractEvaluator):
    """
    Evaluator for computing and plotting the Image-level Confusion Matrix (Authentic vs Tampered).
    """
    def __init__(self, threshold=0.5, class_names=['Authentic', 'Tampered']) -> None:
        """
        Initialize the Confusion Matrix evaluator.
        
        Args:
            threshold (float): Threshold for binary classification.
            class_names (list): Names of the classes for plotting.
        """
        super().__init__()
        self.name = "image_cm" # Short name for logging
        self.desc = "Image-level Confusion Matrix Visualization"
        self.threshold = threshold
        self.class_names = class_names
        # Order: TP, FP, TN, FN
        self.counts = torch.zeros(4, dtype=torch.float64, device='cuda')
        self.output_dir = None

    def batch_update(self, predict_label, label, *args, **kwargs):
        """
        Update the confusion matrix counts for a batch.
        
        Args:
            predict_label (Tensor): Predicted labels (logits or probabilities).
            label (Tensor): Ground truth labels.
        """
        if predict_label is None or label is None:
            return None
            
        predict = (predict_label > self.threshold).float().cuda()
        target = label.float().cuda()
        
        # Ensure same shape
        if predict.shape != target.shape:
             target = target.view_as(predict)

        tp = torch.sum(predict * target)
        tn = torch.sum((1 - predict) * (1 - target))
        fp = torch.sum(predict * (1 - target))
        fn = torch.sum((1 - predict) * target)
        
        self.counts[0] += tp.item()
        self.counts[1] += fp.item()
        self.counts[2] += tn.item()
        self.counts[3] += fn.item()
        
        # Try to capture output_dir from args if available
        if self.output_dir is None and 'args' in kwargs:
            args = kwargs['args']
            # full_log_dir is dataset-specific log dir in test-AniXplore.py
            self.output_dir = getattr(args, 'full_log_dir', getattr(args, 'output_dir', './output_dir'))

        return None

    def epoch_update(self):
        """
        Sync results across GPUs and plot the confusion matrix.
        
        Returns:
            float: Accuracy derived from the confusion matrix.
        """
        t = self.counts.clone()
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            
        tp, fp, tn, fn = t.tolist()
        
        # Confusion matrix: Rows are Actual, Cols are Predicted
        # [[TN, FP],
        #  [FN, TP]]
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plotting
        try:
            plt.figure(figsize=(10, 8))
            sns.set_theme(style="white")
            ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', 
                        xticklabels=self.class_names, yticklabels=self.class_names,
                        annot_kws={"size": 16})
            
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            plt.title('Image-level Confusion Matrix', fontsize=16)
            
            # Add percentage if possible
            total = np.sum(cm)
            if total > 0:
                acc = (tp + tn) / total
            else:
                acc = 0.0

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                save_path = os.path.join(self.output_dir, "confusion_matrix.jpg")
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"[ConfusionMatrix] Plot saved to: {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"[ConfusionMatrix] Error during plotting: {e}")
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        return acc

    def recovery(self):
        """
        Reset counts for the next epoch/dataset.
        """
        self.counts = torch.zeros(4, dtype=torch.float64, device='cuda')

class SourceConfusionMatrix(AbstractEvaluator):
    def __init__(self, num_classes=4, class_names=['A', 'B', 'C', 'D']) -> None:
        super().__init__()
        self.name = "source_cm"
        self.desc = "Source-level Confusion Matrix Visualization"
        self.num_classes = num_classes
        self.class_names = class_names
        self.conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64, device='cuda')
        self.output_dir = None

    def batch_update(self, predict_label, label, *args, **kwargs):
        pred = kwargs.get('predict_source', predict_label)
        target = kwargs.get('source_label', label)
        
        if pred is not None and target is not None:
            for p, t in zip(pred.view(-1), target.view(-1)):
                self.conf_matrix[t.long(), p.long()] += 1

        if self.output_dir is None and 'args' in kwargs:
            args_obj = kwargs['args']
            self.output_dir = getattr(args_obj, 'full_log_dir', getattr(args_obj, 'output_dir', './output_dir'))

        return None

    def epoch_update(self):
        if dist.is_initialized():
            dist.all_reduce(self.conf_matrix, op=dist.ReduceOp.SUM)
            
        cm = self.conf_matrix.cpu().numpy()
        
        try:
            plt.figure(figsize=(10, 8))
            sns.set_theme(style="white")
            ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', 
                        xticklabels=self.class_names, yticklabels=self.class_names,
                        annot_kws={"size": 16})
            
            plt.xlabel('Predicted Source', fontsize=14)
            plt.ylabel('True Source', fontsize=14)
            plt.title('Source-level Confusion Matrix', fontsize=16)
            
            total = np.sum(cm)
            if total > 0:
                acc = np.trace(cm) / total
            else:
                acc = 0.0

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                save_path = os.path.join(self.output_dir, "source_confusion_matrix.jpg")
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"[ConfusionMatrix] Source Plot saved to: {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"[ConfusionMatrix] Error during source plotting: {e}")
            acc = 0.0

        return acc

    def recovery(self):
        self.conf_matrix.zero_()

class PixelConfusionMatrix(AbstractEvaluator):
    def __init__(self, threshold=0.5, class_names=['Authentic Px', 'Tampered Px']):
        super().__init__()
        self.name = "pixel_cm"
        self.desc = "Pixel-level Confusion Matrix Visualization"
        self.threshold = threshold
        self.class_names = class_names
        self.counts = torch.zeros(4, dtype=torch.float64, device='cuda')
        self.output_dir = None

    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        if predict is None or mask is None:
            return None
            
        predict_bin = (predict > self.threshold).float()
        if shape_mask is not None:
            tp = torch.sum(predict_bin * mask * shape_mask)
            tn = torch.sum((1-predict_bin) * (1-mask) * shape_mask)
            fp = torch.sum(predict_bin * (1-mask) * shape_mask)
            fn = torch.sum((1-predict_bin) * mask * shape_mask)
        else:
            tp = torch.sum(predict_bin * mask)
            tn = torch.sum((1-predict_bin) * (1-mask))
            fp = torch.sum(predict_bin * (1-mask))
            fn = torch.sum((1-predict_bin) * mask)

        self.counts[0] += tp.item()
        self.counts[1] += fp.item()
        self.counts[2] += tn.item()
        self.counts[3] += fn.item()

        if self.output_dir is None and 'args' in kwargs:
            args_obj = kwargs['args']
            self.output_dir = getattr(args_obj, 'full_log_dir', getattr(args_obj, 'output_dir', './output_dir'))

        return None

    def epoch_update(self):
        t = self.counts.clone()
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            
        tp, fp, tn, fn = t.tolist()
        cm = np.array([[tn, fp], [fn, tp]])
        
        try:
            plt.figure(figsize=(10, 8))
            sns.set_theme(style="white")
            ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', 
                        xticklabels=self.class_names, yticklabels=self.class_names,
                        annot_kws={"size": 16})
            
            plt.xlabel('Predicted Pixels', fontsize=14)
            plt.ylabel('True Pixels', fontsize=14)
            plt.title('Pixel-level Confusion Matrix', fontsize=16)
            
            total = np.sum(cm)
            if total > 0:
                acc = (tp + tn) / total
            else:
                acc = 0.0

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                save_path = os.path.join(self.output_dir, "pixel_confusion_matrix.jpg")
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"[ConfusionMatrix] Pixel Plot saved to: {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"[ConfusionMatrix] Error during pixel plotting: {e}")
            acc = 0.0

        return acc

    def recovery(self):
        self.counts = torch.zeros(4, dtype=torch.float64, device='cuda')
