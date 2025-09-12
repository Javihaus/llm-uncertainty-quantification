"""
TAP (Theory of Adjacent Possible) Uncertainty Quantification for LLMs
Based on: "Information-Theoretic Uncertainty Quantification for Large Language Models"
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import time


class TAPUncertainty:
    """TAP uncertainty quantification implementation."""
    
    def __init__(self, beta: float = 1.0, alpha: float = 0.9):
        """
        Initialize TAP uncertainty method.
        
        Args:
            beta: Sensitivity parameter for perplexity-to-uncertainty mapping
            alpha: Probability mass threshold for adjacent possible definition
        """
        self.beta = beta
        self.alpha = alpha
    
    def compute_uncertainty(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> Dict[str, float]:
        """
        Compute TAP uncertainty for a sequence.
        
        Args:
            logits: Model logits [seq_len, vocab_size]
            target_tokens: Target token ids [seq_len]
            
        Returns:
            Dictionary with uncertainty measures
        """
        start_time = time.time()
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Compute perplexity for each token
        token_probs = probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        perplexities = 1.0 / token_probs
        
        # TAP uncertainty using exponential mapping
        token_uncertainties = 1.0 - torch.exp(-self.beta * perplexities)
        
        # Sequence-level uncertainty (mean)
        sequence_uncertainty = token_uncertainties.mean().item()
        
        # Additional measures
        entropy = self._compute_entropy(probs)
        adjacent_possible_size = self._compute_adjacent_possible_size(probs)
        
        computation_time = time.time() - start_time
        
        return {
            'tap_uncertainty': sequence_uncertainty,
            'mean_perplexity': perplexities.mean().item(),
            'mean_entropy': entropy.mean().item(),
            'adjacent_possible_size': adjacent_possible_size.mean().item(),
            'computation_time': computation_time
        }
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distributions."""
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
    
    def _compute_adjacent_possible_size(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute size of adjacent possible (number of tokens above threshold)."""
        # Sort probabilities and find cumulative sum
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        
        # Find threshold that captures alpha fraction of probability mass
        threshold_idx = torch.argmax((cumsum >= self.alpha).float(), dim=-1)
        threshold_probs = sorted_probs.gather(-1, threshold_idx.unsqueeze(-1))
        
        # Count tokens above threshold
        above_threshold = (probs >= threshold_probs.unsqueeze(-1)).sum(dim=-1)
        
        return above_threshold.float()


class BaselineUncertaintyMethods:
    """Baseline uncertainty quantification methods for comparison."""
    
    @staticmethod
    def softmax_confidence(logits: torch.Tensor, target_tokens: torch.Tensor) -> Dict[str, float]:
        """Maximum softmax probability as uncertainty measure."""
        start_time = time.time()
        
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        uncertainty = 1.0 - max_probs.mean().item()
        
        computation_time = time.time() - start_time
        
        return {
            'softmax_uncertainty': uncertainty,
            'computation_time': computation_time
        }
    
    @staticmethod
    def entropy_uncertainty(logits: torch.Tensor, target_tokens: torch.Tensor) -> Dict[str, float]:
        """Entropy-based uncertainty."""
        start_time = time.time()
        
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Normalize by log(vocab_size)
        vocab_size = logits.size(-1)
        normalized_entropy = entropy / np.log(vocab_size)
        uncertainty = normalized_entropy.mean().item()
        
        computation_time = time.time() - start_time
        
        return {
            'entropy_uncertainty': uncertainty,
            'computation_time': computation_time
        }
    
    @staticmethod
    def predictive_entropy(logits: torch.Tensor, target_tokens: torch.Tensor) -> Dict[str, float]:
        """Token-level predictive entropy."""
        start_time = time.time()
        
        probs = torch.softmax(logits, dim=-1)
        target_probs = probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        surprisal = -torch.log(target_probs + 1e-10)
        uncertainty = surprisal.mean().item()
        
        computation_time = time.time() - start_time
        
        return {
            'predictive_entropy': uncertainty,
            'computation_time': computation_time
        }


class UncertaintyEvaluator:
    """Evaluate uncertainty methods using various metrics."""
    
    def __init__(self):
        self.tap = TAPUncertainty()
        self.baselines = BaselineUncertaintyMethods()
    
    def evaluate_all_methods(self, logits: torch.Tensor, target_tokens: torch.Tensor, 
                           correct_predictions: torch.Tensor) -> Dict[str, Dict]:
        """
        Evaluate all uncertainty methods.
        
        Args:
            logits: Model logits [seq_len, vocab_size]
            target_tokens: Target token ids [seq_len]
            correct_predictions: Binary tensor indicating correct predictions [seq_len]
            
        Returns:
            Dictionary with results from all methods
        """
        results = {}
        
        # TAP method
        results['tap'] = self.tap.compute_uncertainty(logits, target_tokens)
        
        # Baseline methods
        results['softmax'] = self.baselines.softmax_confidence(logits, target_tokens)
        results['entropy'] = self.baselines.entropy_uncertainty(logits, target_tokens)
        results['predictive_entropy'] = self.baselines.predictive_entropy(logits, target_tokens)
        
        # Add ground truth accuracy for calibration evaluation
        accuracy = correct_predictions.float().mean().item()
        for method in results:
            results[method]['accuracy'] = accuracy
        
        return results