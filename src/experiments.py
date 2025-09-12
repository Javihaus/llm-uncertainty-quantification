"""
Main experiments orchestration for TAP uncertainty quantification validation.
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
import logging

from models import ModelFactory
from datasets import DatasetManager
from uncertainty_methods import UncertaintyEvaluator
from metrics import CalibrationMetrics, UncertaintyMetrics, ComputationalMetrics, ResultsAggregator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run uncertainty quantification experiments."""
    
    def __init__(self, output_dir: str = "results/", device: str = None):
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        self.dataset_manager = DatasetManager()
        self.evaluator = UncertaintyEvaluator()
        self.aggregator = ResultsAggregator()
        
        logger.info(f"Using device: {self.device}")
    
    def run_full_experiment(self, 
                          models: List[str] = None,
                          datasets: List[str] = None,
                          num_samples: int = 50) -> Dict:
        """Run full experimental evaluation."""
        
        # Default models (lightweight ones for CPU)
        if models is None:
            models = ['gpt2']  # Start with just GPT-2 for testing
        
        # Default datasets
        if datasets is None:
            datasets = ['factual', 'truthfulqa']
        
        logger.info(f"Running experiments with models: {models}")
        logger.info(f"Datasets: {datasets}")
        
        # Load datasets
        all_datasets = self.dataset_manager.load_all_datasets(
            truthful_qa_samples=num_samples,
            mmlu_samples_per_subject=min(num_samples, 20),
            factual_samples=num_samples
        )
        
        # Run experiments for each model-dataset combination
        results = {}
        
        for model_name in models:
            logger.info(f"\\n=== Evaluating {model_name} ===")
            
            try:
                # Load model
                model_interface = ModelFactory.create_model(model_name, self.device)
                model_results = {}
                
                for dataset_name in datasets:
                    if dataset_name not in all_datasets:
                        logger.warning(f"Dataset {dataset_name} not found, skipping...")
                        continue
                    
                    logger.info(f"\\nEvaluating on {dataset_name} dataset...")
                    dataset = all_datasets[dataset_name][:num_samples]  # Limit samples
                    
                    # Run evaluation on this dataset
                    dataset_results = self._evaluate_on_dataset(
                        model_interface, dataset, dataset_name
                    )
                    
                    model_results[dataset_name] = dataset_results
                    
                    # Add results to aggregator
                    for method, metrics in dataset_results.items():
                        self.aggregator.add_result(model_name, dataset_name, method, metrics)
                
                results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Save results
        self._save_results(results)
        
        # Create summary
        summary = self._create_summary(results)
        
        return {
            'detailed_results': results,
            'summary': summary
        }
    
    def _evaluate_on_dataset(self, model_interface, dataset: List[Dict], 
                           dataset_name: str) -> Dict:
        """Evaluate uncertainty methods on a single dataset."""
        
        all_uncertainties = {method: [] for method in ['tap', 'softmax', 'entropy', 'predictive_entropy']}
        all_accuracies = []
        all_computation_times = {method: [] for method in all_uncertainties.keys()}
        
        logger.info(f"Processing {len(dataset)} samples...")
        
        for item in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            try:
                # Get model predictions
                if item['type'] == 'multiple_choice':
                    results = self._evaluate_multiple_choice(model_interface, item)
                else:
                    results = self._evaluate_open_ended(model_interface, item)
                
                if results is None:
                    continue
                
                # Extract uncertainty scores and accuracy
                uncertainty_results, accuracy = results
                
                for method in all_uncertainties.keys():
                    if method in uncertainty_results:
                        all_uncertainties[method].append(uncertainty_results[method]['uncertainty'])
                        all_computation_times[method].append(
                            uncertainty_results[method].get('computation_time', 0)
                        )
                
                all_accuracies.append(accuracy)
                
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
        
        # Compute evaluation metrics for each method
        method_results = {}
        accuracies = np.array(all_accuracies)
        
        for method in all_uncertainties.keys():
            if all_uncertainties[method]:
                uncertainties = np.array(all_uncertainties[method])
                
                # Calibration metrics
                ece_results = CalibrationMetrics.expected_calibration_error(uncertainties, accuracies)
                brier_score = CalibrationMetrics.brier_score(uncertainties, accuracies)
                
                # Uncertainty-accuracy correlation
                corr_results = UncertaintyMetrics.uncertainty_accuracy_correlation(uncertainties, accuracies)
                
                # AUROC for error prediction
                auroc = UncertaintyMetrics.auroc_uncertainty(uncertainties, accuracies)
                
                # Computational efficiency
                comp_times = all_computation_times[method]
                
                method_results[method] = {
                    'ece': ece_results['ece'],
                    'mce': ece_results['mce'],
                    'brier_score': brier_score,
                    'pearson_correlation': corr_results['pearson_correlation'],
                    'spearman_correlation': corr_results['spearman_correlation'],
                    'auroc': auroc,
                    'mean_computation_time': np.mean(comp_times),
                    'std_computation_time': np.std(comp_times),
                    'num_samples': len(uncertainties),
                    'mean_accuracy': accuracies.mean()
                }
        
        return method_results
    
    def _evaluate_multiple_choice(self, model_interface, item: Dict) -> Optional[tuple]:
        """Evaluate multiple choice question."""
        question = item['question']
        choices = item['choices']
        correct_idx = item['correct_idx']
        
        # Create prompts for each choice
        choice_scores = []
        all_uncertainty_results = []
        
        for i, choice in enumerate(choices):
            prompt = f"Question: {question}\\nAnswer: {choice}"
            
            try:
                # Get logits for this choice
                logits, token_ids = model_interface.get_logits(prompt)
                
                if logits.size(0) == 0:
                    continue
                
                # Create binary correctness indicator
                correct_predictions = torch.ones(token_ids.size(0)) if i == correct_idx else torch.zeros(token_ids.size(0))
                
                # Evaluate uncertainty methods
                uncertainty_results = self.evaluator.evaluate_all_methods(
                    logits, token_ids, correct_predictions
                )
                
                all_uncertainty_results.append(uncertainty_results)
                
                # Score this choice (average log probability)
                token_probs = torch.softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
                choice_scores.append(torch.log(token_probs + 1e-10).mean().item())
                
            except Exception as e:
                logger.warning(f"Error evaluating choice {i}: {e}")
                continue
        
        if not choice_scores or not all_uncertainty_results:
            return None
        
        # Determine predicted choice and accuracy
        predicted_idx = np.argmax(choice_scores)
        accuracy = 1.0 if predicted_idx == correct_idx else 0.0
        
        # Use uncertainty from predicted choice
        if predicted_idx < len(all_uncertainty_results):
            uncertainty_results = all_uncertainty_results[predicted_idx]
            
            # Reformat uncertainty results
            formatted_results = {}
            for method in uncertainty_results:
                method_data = uncertainty_results[method]
                if 'tap_uncertainty' in method_data:
                    formatted_results['tap'] = {
                        'uncertainty': method_data['tap_uncertainty'],
                        'computation_time': method_data['computation_time']
                    }
                elif 'softmax_uncertainty' in method_data:
                    formatted_results['softmax'] = {
                        'uncertainty': method_data['softmax_uncertainty'],
                        'computation_time': method_data['computation_time']
                    }
                elif 'entropy_uncertainty' in method_data:
                    formatted_results['entropy'] = {
                        'uncertainty': method_data['entropy_uncertainty'],
                        'computation_time': method_data['computation_time']
                    }
                elif 'predictive_entropy' in method_data:
                    formatted_results['predictive_entropy'] = {
                        'uncertainty': method_data['predictive_entropy'],
                        'computation_time': method_data['computation_time']
                    }
            
            return formatted_results, accuracy
        
        return None
    
    def _evaluate_open_ended(self, model_interface, item: Dict) -> Optional[tuple]:
        """Evaluate open-ended question."""
        prompt = item['prompt']
        correct_answer = item['correct_answer'].lower()
        
        try:
            # Generate response
            generation_result = model_interface.generate_with_logits(prompt, max_length=20)
            
            generated_text = generation_result['generated_text'].lower()
            logits = generation_result['logits']
            token_ids = generation_result['generated_ids']
            
            if logits.size(0) == 0:
                return None
            
            # Simple accuracy check (substring match)
            accuracy = 1.0 if correct_answer in generated_text else 0.0
            correct_predictions = torch.ones(token_ids.size(0)) * accuracy
            
            # Evaluate uncertainty methods
            uncertainty_results = self.evaluator.evaluate_all_methods(
                logits, token_ids, correct_predictions
            )
            
            # Reformat results
            formatted_results = {}
            for method in uncertainty_results:
                method_data = uncertainty_results[method]
                if 'tap_uncertainty' in method_data:
                    formatted_results['tap'] = {
                        'uncertainty': method_data['tap_uncertainty'],
                        'computation_time': method_data['computation_time']
                    }
                elif 'softmax_uncertainty' in method_data:
                    formatted_results['softmax'] = {
                        'uncertainty': method_data['softmax_uncertainty'],
                        'computation_time': method_data['computation_time']
                    }
                elif 'entropy_uncertainty' in method_data:
                    formatted_results['entropy'] = {
                        'uncertainty': method_data['entropy_uncertainty'],
                        'computation_time': method_data['computation_time']
                    }
                elif 'predictive_entropy' in method_data:
                    formatted_results['predictive_entropy'] = {
                        'uncertainty': method_data['predictive_entropy'],
                        'computation_time': method_data['computation_time']
                    }
            
            return formatted_results, accuracy
            
        except Exception as e:
            logger.warning(f"Error evaluating open-ended question: {e}")
            return None
    
    def _save_results(self, results: Dict):
        """Save detailed results to JSON."""
        output_file = os.path.join(self.output_dir, 'detailed_results.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = deep_convert(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {output_file}")
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create summary of experimental results."""
        summary = {
            'experiment_info': {
                'device': self.device,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'models_evaluated': list(results.keys())
            },
            'performance_summary': {},
            'efficiency_summary': {}
        }
        
        # Performance summary
        for model in results:
            model_summary = {}
            for dataset in results[model]:
                dataset_summary = {}
                for method in results[model][dataset]:
                    method_data = results[model][dataset][method]
                    dataset_summary[method] = {
                        'ECE': round(method_data.get('ece', 0), 4),
                        'Brier Score': round(method_data.get('brier_score', 0), 4),
                        'AUROC': round(method_data.get('auroc', 0), 4),
                        'Accuracy': round(method_data.get('mean_accuracy', 0), 4)
                    }
                model_summary[dataset] = dataset_summary
            summary['performance_summary'][model] = model_summary
        
        # Efficiency summary (computation times)
        for model in results:
            model_times = {}
            for dataset in results[model]:
                dataset_times = {}
                for method in results[model][dataset]:
                    method_data = results[model][dataset][method]
                    dataset_times[method] = {
                        'Mean Time (s)': round(method_data.get('mean_computation_time', 0), 6),
                        'Std Time (s)': round(method_data.get('std_computation_time', 0), 6)
                    }
                model_times[dataset] = dataset_times
            summary['efficiency_summary'][model] = model_times
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")
        
        # Print summary to console
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print summary to console."""
        print("\\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print(f"\\nModels evaluated: {', '.join(summary['experiment_info']['models_evaluated'])}")
        print(f"Timestamp: {summary['experiment_info']['timestamp']}")
        print(f"Device: {summary['experiment_info']['device']}")
        
        print("\\n" + "-"*40)
        print("PERFORMANCE RESULTS")
        print("-"*40)
        
        for model in summary['performance_summary']:
            print(f"\\n{model.upper()}:")
            for dataset in summary['performance_summary'][model]:
                print(f"  {dataset}:")
                methods = summary['performance_summary'][model][dataset]
                
                # Create table
                method_names = list(methods.keys())
                if method_names:
                    metrics = ['ECE', 'Brier Score', 'AUROC', 'Accuracy']
                    
                    # Print header
                    print(f"    {'Method':<15} {'ECE':<8} {'Brier':<8} {'AUROC':<8} {'Acc':<8}")
                    print(f"    {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                    
                    # Print data
                    for method in method_names:
                        data = methods[method]
                        print(f"    {method:<15} {data['ECE']:<8.4f} {data['Brier Score']:<8.4f} "
                              f"{data['AUROC']:<8.4f} {data['Accuracy']:<8.4f}")
        
        print("\\n" + "-"*40)
        print("COMPUTATIONAL EFFICIENCY")
        print("-"*40)
        
        for model in summary['efficiency_summary']:
            print(f"\\n{model.upper()} - Mean computation time per sample:")
            for dataset in summary['efficiency_summary'][model]:
                print(f"  {dataset}:")
                methods = summary['efficiency_summary'][model][dataset]
                for method, times in methods.items():
                    mean_time = times['Mean Time (s)']
                    print(f"    {method}: {mean_time:.6f}s")


def main():
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description='Run TAP uncertainty quantification experiments')
    parser.add_argument('--models', nargs='+', default=['gpt2'], 
                        help='Models to evaluate')
    parser.add_argument('--datasets', nargs='+', default=['factual', 'truthfulqa'],
                        help='Datasets to use')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples per dataset')
    parser.add_argument('--output-dir', default='results/',
                        help='Output directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Device to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Run experiments
    runner = ExperimentRunner(args.output_dir, args.device)
    results = runner.run_full_experiment(
        models=args.models,
        datasets=args.datasets,
        num_samples=args.num_samples
    )
    
    print("\\nExperiment completed successfully!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()