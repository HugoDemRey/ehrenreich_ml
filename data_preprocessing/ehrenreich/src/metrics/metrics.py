import numpy as np

class TS_Evaluator():
    def __init__(self, tolerance_seconds=0.5):
        self.tolerance_seconds = tolerance_seconds

    def true_positives(self, y_true, y_pred):
        """
        Count true positives: each true boundary can be matched by at most one prediction within tolerance.
        """
        matched_pred = set()
        num_tp = 0

        for true_time in y_true:
            # Find all unmatched predictions within tolerance
            candidates = [i for i, pred_time in enumerate(y_pred)
                        if i not in matched_pred and abs(pred_time - true_time) <= self.tolerance_seconds]
            if candidates:
                # Match the closest prediction (optional, but standard)
                closest_idx = min(candidates, key=lambda i: abs(y_pred[i] - true_time))
                matched_pred.add(closest_idx)
                num_tp += 1

        return num_tp

    def false_positives(self, y_true, y_pred):
        """
        Count false positives: predicted boundaries that don't match any reference boundary within tolerance
        
        Args:
            y_true (list): Reference boundary timestamps
            y_pred (list): Predicted boundary timestamps
            
        Returns:
            int: Number of false positives
        """
        num_fp = 0
        
        for pred_time in y_pred:
            # Check if this prediction is NOT within tolerance of any reference boundary
            is_match = False
            for true_time in y_true:
                if abs(pred_time - true_time) <= self.tolerance_seconds:
                    is_match = True
                    break
            
            if not is_match:
                num_fp += 1
        
        return num_fp

    def false_negatives(self, y_true, y_pred):
        """
        Count false negatives: reference boundaries that don't have any prediction within tolerance
        
        Args:
            y_true (list): Reference boundary timestamps
            y_pred (list): Predicted boundary timestamps
            
        Returns:
            int: Number of false negatives
        """
        num_fn = 0
        
        for true_time in y_true:
            # Check if this reference boundary has NO prediction within tolerance
            is_detected = False
            for pred_time in y_pred:
                if abs(pred_time - true_time) <= self.tolerance_seconds:
                    is_detected = True
                    break
            
            if not is_detected:
                num_fn += 1
        
        return num_fn

    def precision(self, y_true, y_pred):
        """
        Calculate precision: TP / (TP + FP)
        
        Args:
            y_true (list): Reference boundary timestamps
            y_pred (list): Predicted boundary timestamps
            
        Returns:
            float: Precision score
        """
        num_tp = self.true_positives(y_true, y_pred)
        num_fp = self.false_positives(y_true, y_pred)
        
        if num_tp + num_fp == 0:
            return 0.0
        
        return num_tp / (num_tp + num_fp)

    def recall(self, y_true, y_pred):
        """
        Calculate recall: TP / (TP + FN)
        
        Args:
            y_true (list): Reference boundary timestamps
            y_pred (list): Predicted boundary timestamps
            
        Returns:
            float: Recall score
        """
        num_tp = self.true_positives(y_true, y_pred)
        num_fn = self.false_negatives(y_true, y_pred)
        
        if num_tp + num_fn == 0:
            return 0.0
        
        return num_tp / (num_tp + num_fn)

    def f1_score(self, y_true, y_pred):
        """
        Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        
        Args:
            y_true (list): Reference boundary timestamps
            y_pred (list): Predicted boundary timestamps
            
        Returns:
            float: F1 score
        """
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)

    def evaluate(self, y_true, y_pred):
        """
        Comprehensive evaluation returning all metrics
        
        Args:
            y_true (list): Reference boundary timestamps
            y_pred (list): Predicted boundary timestamps
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        num_tp = self.true_positives(y_true, y_pred)
        num_fp = self.false_positives(y_true, y_pred)
        num_fn = self.false_negatives(y_true, y_pred)
        
        precision, recall, f1 = self.precision(y_true, y_pred), self.recall(y_true, y_pred), self.f1_score(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': num_tp,
            'false_positives': num_fp,
            'false_negatives': num_fn,
            'tolerance_seconds': self.tolerance_seconds
        }
    
    def plot_evaluation(self, y_true, y_pred, save_path=None):
        """
        Plot evaluation results with tolerance bands and TP/FP classification
        
        Args:
            y_true (list): Reference boundary timestamps
            y_pred (list): Predicted boundary timestamps
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot tolerance bands around true boundaries
        for true_time in y_true:
            ax.axvspan(true_time - self.tolerance_seconds, 
                      true_time + self.tolerance_seconds, 
                      alpha=0.2, color='blue', 
                      label='Tolerance Band' if true_time == y_true[0] else "")
        
        # Plot true boundaries as dark vertical lines
        ax.eventplot(y_true, lineoffsets=2, colors='black', linewidths=3, 
                    linelengths=0.3, label='True Boundaries')
        
        # Classify predictions as TP or FP and plot accordingly
        tp_predictions = []
        fp_predictions = []
        
        for pred_time in y_pred:
            is_tp = False
            # Check if this prediction is within tolerance of any true boundary
            for true_time in y_true:
                if abs(pred_time - true_time) <= self.tolerance_seconds:
                    is_tp = True
                    break
            
            if is_tp:
                tp_predictions.append(pred_time)
            else:
                fp_predictions.append(pred_time)
        
        # Plot True Positives as green lines
        if tp_predictions:
            ax.eventplot(tp_predictions, lineoffsets=1, colors='green', linewidths=2,
                        linelengths=0.4, label=f'True Positives ({len(tp_predictions)})')
        
        # Plot False Positives as red lines  
        if fp_predictions:
            ax.eventplot(fp_predictions, lineoffsets=0.5, colors='red', linewidths=2,
                        linelengths=0.4, label=f'False Positives ({len(fp_predictions)})')
        
        # Calculate and display metrics
        metrics = self.evaluate(y_true, y_pred)
        num_fn = metrics['false_negatives']
        
        # Set plot properties
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0.5, 1, 2], ['FP', 'TP', 'True'])
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Boundary Detection Evaluation (Tolerance: ±{self.tolerance_seconds}s)\n'
                    f'P: {metrics["precision"]:.3f}, R: {metrics["recall"]:.3f}, '
                    f'F1: {metrics["f1_score"]:.3f}, FN: {num_fn}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()