from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
import numpy as np
from typing import Dict, List

class Evaluator:
    @staticmethod
    def compute_metrics(y_true: List[int], y_scores: List[float]) -> Dict[str, float]:
        """
        Computes ROC-AUC and TPR at specific FPR thresholds (0.01%, 0.1%, 1%).
        y_true: 0 for human, 1 for machine
        y_scores: Higher scores indicate machine-generated (Binoculars score).
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # ROC AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        metrics = {'ROC-AUC': roc_auc}
        
        # TPR at fixed FPRs
        target_fprs = [0.0001, 0.001, 0.01] # 0.01%, 0.1%, 1%
        
        for target_fpr in target_fprs:
            # Find the index where FPR is closest to target_fpr but <= target_fpr
            # Since fpr is increasing, we can use searchsorted or manual loop
            # We want the TPR corresponding to the largest FPR <= target_fpr
            
            # Simple approach: interpolate
            tpr_at_fpr = np.interp(target_fpr, fpr, tpr)
            metrics[f'TPR@{target_fpr*100}%FPR'] = tpr_at_fpr

        return metrics
