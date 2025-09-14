#!/usr/bin/env python3
import numpy as np
import time
import json
import os
from datetime import datetime

# Simple PBA implementation for validation
class SimplePBA:
    def __init__(self, beta=0.5):
        self.beta = beta
    
    def compute_uncertainty(self, logits, target_tokens):
        """Compute PBA uncertainty."""
        # Convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Get target probabilities
        target_probs = probs[np.arange(len(target_tokens)), target_tokens]
        
        # Compute perplexity
        perplexities = 1.0 / (target_probs + 1e-10)
        
        # PBA uncertainty
        uncertainties = 1.0 - np.exp(-self.beta * perplexities)
        
        # Entropy
        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs, axis=-1)
        
        return {
            'pba_uncertainty': np.mean(uncertainties),
            'mean_perplexity': np.mean(perplexities),
            'mean_entropy': np.mean(entropy),
            'individual_uncertainties': uncertainties.tolist(),
            'individual_perplexities': perplexities.tolist()
        }

def create_test_cases():
    """Create test cases with different confidence levels."""
    vocab_size = 1000
    seq_len = 10
    
    # High confidence: peaked distribution
    high_conf_logits = np.random.normal(0, 0.1, (seq_len, vocab_size))
    high_conf_logits[:, 0] = 8.0  # Token 0 is very likely
    high_conf_targets = np.zeros(seq_len, dtype=int)
    
    # Low confidence: flat distribution  
    low_conf_logits = np.random.normal(0, 0.5, (seq_len, vocab_size))
    low_conf_targets = np.random.randint(0, vocab_size, seq_len)
    
    # Medium confidence: moderate peak
    med_conf_logits = np.random.normal(0, 1.0, (seq_len, vocab_size))
    med_conf_logits[:, :50] += 2.0  # Top 50 tokens more likely
    med_conf_targets = np.random.randint(0, 50, seq_len)
    
    return [
        ('High Confidence', high_conf_logits, high_conf_targets),
        ('Low Confidence', low_conf_logits, low_conf_targets),
        ('Medium Confidence', med_conf_logits, med_conf_targets)
    ]

def run_validation():
    """Run PBA validation experiment."""
    print("PBA Uncertainty Quantification - Validation Experiment")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    
    pba = SimplePBA(beta=0.5)
    test_cases = create_test_cases()
    results = {}
    
    for name, logits, targets in test_cases:
        print(f"\nTesting: {name}")
        print("-" * 30)
        
        start_time = time.time()
        result = pba.compute_uncertainty(logits, targets)
        computation_time = time.time() - start_time
        
        print(f"PBA Uncertainty:  {result['pba_uncertainty']:.4f}")
        print(f"Mean Perplexity:  {result['mean_perplexity']:.4f}")
        print(f"Mean Entropy:     {result['mean_entropy']:.4f}")
        print(f"Computation Time: {computation_time:.6f}s")
        
        results[name] = {
            **result,
            'computation_time': computation_time
        }
    
    # Validation checks
    print("\nValidation Results:")
    print("-" * 30)
    
    high_conf = results['High Confidence']
    low_conf = results['Low Confidence']
    med_conf = results['Medium Confidence']
    
    checks = [
        ("Low conf > High conf uncertainty", 
         low_conf['pba_uncertainty'] > high_conf['pba_uncertainty']),
        ("Low conf > High conf perplexity", 
         low_conf['mean_perplexity'] > high_conf['mean_perplexity']),
        ("Low conf > High conf entropy", 
         low_conf['mean_entropy'] > high_conf['mean_entropy']),
        ("Medium between high and low", 
         high_conf['pba_uncertainty'] < med_conf['pba_uncertainty'] < low_conf['pba_uncertainty']),
        ("Fast computation (< 1ms)", 
         all(r['computation_time'] < 0.001 for r in results.values()))
    ]
    
    all_passed = True
    for desc, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {desc}")
        if not passed:
            all_passed = False
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'validation_checks': {desc: passed for desc, passed in checks},
        'all_checks_passed': all_passed
    }
    
    with open('results/pba_validation.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: results/pba_validation.json")
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("PBA implementation is working correctly.")
        
        # Create simple summary
        summary = {
            'method': 'PBA (Theory of Adjacent Possible)',
            'validation_status': 'PASSED',
            'key_findings': [
                f"PBA uncertainty correctly increases with model uncertainty",
                f"High confidence case: {high_conf['pba_uncertainty']:.4f}",
                f"Low confidence case: {low_conf['pba_uncertainty']:.4f}",
                f"Computation time: ~{np.mean([r['computation_time'] for r in results.values()])*1000:.2f}μs per sample"
            ],
            'next_steps': [
                'Ready for full model evaluation',
                'Install transformers library',
                'Run experiments with real LLMs'
            ]
        }
        
        with open('results/validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\nValidation Summary:")
        for finding in summary['key_findings']:
            print(f"• {finding}")
        
    else:
        print("\n⚠️ SOME VALIDATIONS FAILED")
        print("Check implementation before proceeding.")
    
    return results, all_passed

if __name__ == '__main__':
    run_validation()