"""
Evaluation metrics for uncertainty quantification.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns


class CalibrationMetrics:
    """Calibration evaluation metrics."""
    
    @staticmethod
    def expected_calibration_error(uncertainties: np.ndarray, accuracies: np.ndarray, 
                                 n_bins: int = 10) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            uncertainties: Uncertainty scores [0, 1]
            accuracies: Binary accuracy scores {0, 1}
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with ECE and related metrics
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0  # Maximum Calibration Error
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_uncertainty_in_bin = uncertainties[in_bin].mean()
                
                calibration_error = abs(avg_uncertainty_in_bin - accuracy_in_bin)
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
                
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'prop_in_bin': prop_in_bin,
                    'accuracy_in_bin': accuracy_in_bin,
                    'avg_uncertainty_in_bin': avg_uncertainty_in_bin,
                    'calibration_error': calibration_error
                })
        
        return {
            'ece': ece,
            'mce': mce,
            'bin_data': bin_data
        }
    
    @staticmethod
    def brier_score(uncertainties: np.ndarray, accuracies: np.ndarray) -> float:
        """
        Compute Brier Score.
        
        Args:
            uncertainties: Uncertainty scores [0, 1]
            accuracies: Binary accuracy scores {0, 1}
            
        Returns:
            Brier score (lower is better)
        """
        # Convert uncertainty to confidence
        confidences = 1 - uncertainties
        return brier_score_loss(accuracies, confidences)
    
    @staticmethod
    def reliability_diagram(uncertainties: np.ndarray, accuracies: np.ndarray, 
                          n_bins: int = 10, save_path: str = None) -> plt.Figure:
        """
        Create reliability diagram.
        
        Args:
            uncertainties: Uncertainty scores [0, 1]
            accuracies: Binary accuracy scores {0, 1}
            n_bins: Number of bins
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        ece_data = CalibrationMetrics.expected_calibration_error(
            uncertainties, accuracies, n_bins
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        # Calibration bars
        bin_data = ece_data['bin_data']
        if bin_data:
            bin_centers = [(d['bin_lower'] + d['bin_upper']) / 2 for d in bin_data]
            accuracies_in_bins = [d['accuracy_in_bin'] for d in bin_data]
            confidences_in_bins = [1 - d['avg_uncertainty_in_bin'] for d in bin_data]
            bin_sizes = [d['prop_in_bin'] for d in bin_data]
            
            # Scale bin widths by proportion of data
            bar_width = 0.08
            scaled_widths = [bar_width * size * len(bin_data) for size in bin_sizes]
            
            bars = ax.bar(bin_centers, accuracies_in_bins, width=scaled_widths, 
                         alpha=0.7, edgecolor='black', label='Accuracy')
            
            # Add confidence points
            ax.scatter(confidences_in_bins, accuracies_in_bins, 
                      s=50, color='red', zorder=3, label='Avg Confidence')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy') 
        ax.set_title(f'Reliability Diagram (ECE: {ece_data["ece"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class UncertaintyMetrics:
    """Additional uncertainty evaluation metrics."""
    
    @staticmethod
    def uncertainty_accuracy_correlation(uncertainties: np.ndarray, 
                                       accuracies: np.ndarray) -> Dict[str, float]:
        """
        Compute correlation between uncertainty and accuracy.
        
        Args:
            uncertainties: Uncertainty scores
            accuracies: Binary accuracy scores
            
        Returns:
            Correlation metrics
        """
        # Pearson correlation
        pearson_corr = np.corrcoef(uncertainties, accuracies)[0, 1]
        
        # Spearman correlation
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(uncertainties, accuracies)
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        }
    
    @staticmethod
    def auroc_uncertainty(uncertainties: np.ndarray, accuracies: np.ndarray) -> float:
        """
        Compute AUROC for uncertainty as predictor of errors.
        
        Args:
            uncertainties: Uncertainty scores
            accuracies: Binary accuracy scores (1=correct, 0=incorrect)
            
        Returns:
            AUROC score
        """
        from sklearn.metrics import roc_auc_score
        
        # Convert to error prediction task (1=error, 0=correct)
        errors = 1 - accuracies
        
        # High uncertainty should predict errors
        return roc_auc_score(errors, uncertainties)
    
    @staticmethod
    def uncertainty_quantiles(uncertainties: np.ndarray, 
                            accuracies: np.ndarray) -> Dict[str, float]:
        """
        Analyze accuracy at different uncertainty quantiles.
        
        Args:
            uncertainties: Uncertainty scores
            accuracies: Binary accuracy scores
            
        Returns:
            Accuracy at different uncertainty levels
        """
        results = {}
        
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            threshold = np.quantile(uncertainties, q)
            high_uncertainty_mask = uncertainties >= threshold
            
            if high_uncertainty_mask.sum() > 0:
                acc_high_uncertainty = accuracies[high_uncertainty_mask].mean()
                results[f'accuracy_above_{q}_quantile'] = acc_high_uncertainty
        
        return results


class ComputationalMetrics:
    """Metrics for computational efficiency."""
    
    @staticmethod
    def efficiency_comparison(method_times: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Compare computational efficiency across methods.
        
        Args:
            method_times: Dictionary mapping method names to lists of computation times
            
        Returns:
            Efficiency statistics for each method
        """
        results = {}
        
        for method, times in method_times.items():
            times_array = np.array(times)
            results[method] = {
                'mean_time': times_array.mean(),
                'std_time': times_array.std(),
                'median_time': np.median(times_array),
                'total_time': times_array.sum()
            }
        
        return results
    
    @staticmethod
    def speedup_analysis(baseline_times: List[float], 
                        method_times: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute speedup relative to baseline.
        
        Args:
            baseline_times: Computation times for baseline method
            method_times: Times for other methods
            
        Returns:
            Speedup factors
        """
        baseline_mean = np.mean(baseline_times)
        speedups = {}
        
        for method, times in method_times.items():
            method_mean = np.mean(times)
            speedups[method] = baseline_mean / method_mean
        
        return speedups


class ResultsAggregator:
    """Aggregate and summarize experimental results."""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, model: str, dataset: str, method: str, metrics: Dict):
        """Add experimental result."""
        if model not in self.results:
            self.results[model] = {}
        if dataset not in self.results[model]:
            self.results[model][dataset] = {}
        
        self.results[model][dataset][method] = metrics
    
    def get_summary_table(self) -> Dict:
        """Create summary table of all results."""
        summary = {}
        
        for model in self.results:
            summary[model] = {}
            for dataset in self.results[model]:
                summary[model][dataset] = {}
                
                methods = self.results[model][dataset]
                for method in methods:
                    metrics = methods[method]
                    summary[model][dataset][method] = {
                        'ece': metrics.get('ece', np.nan),
                        'brier_score': metrics.get('brier_score', np.nan),
                        'computation_time': metrics.get('mean_computation_time', np.nan),
                        'auroc': metrics.get('auroc', np.nan)
                    }
        
        return summary
    
    def create_comparison_plots(self, save_dir: str = 'results/plots/'):
        """Create comparison plots across methods and models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # ECE comparison
        self._plot_metric_comparison('ece', 'Expected Calibration Error', 
                                   f'{save_dir}/ece_comparison.png')
        
        # Brier score comparison  
        self._plot_metric_comparison('brier_score', 'Brier Score',
                                   f'{save_dir}/brier_comparison.png')
        
        # Computation time comparison
        self._plot_metric_comparison('computation_time', 'Computation Time (s)',
                                   f'{save_dir}/time_comparison.png')
    
    def _plot_metric_comparison(self, metric: str, metric_name: str, save_path: str):
        """Plot comparison of specific metric across methods."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(self.results.keys())
        methods = set()
        for model in models:
            for dataset in self.results[model]:
                methods.update(self.results[model][dataset].keys())
        
        methods = list(methods)
        
        # Create grouped bar plot
        x = np.arange(len(models))
        width = 0.2
        
        for i, method in enumerate(methods):
            values = []
            for model in models:
                # Average across datasets for this model-method combination
                model_values = []
                for dataset in self.results[model]:
                    if method in self.results[model][dataset]:
                        val = self.results[model][dataset][method].get(metric, np.nan)
                        if not np.isnan(val):
                            model_values.append(val)
                
                values.append(np.mean(model_values) if model_values else np.nan)
            
            ax.bar(x + i * width, values, width, label=method, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison Across Models')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([m.split('/')[-1] if '/' in m else m for m in models])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()