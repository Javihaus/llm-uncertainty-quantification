# PBA Uncertainty Analysis

## Quick Start

To replicate the experiments:

```bash
# Install dependencies
pip install torch transformers datasets numpy pandas scikit-learn matplotlib

# Run basic validation
python3 validate_pba.py

# Run full experiments (requires transformers library)
python3 src/experiments.py --models gpt2 --datasets factual truthfulqa --num-samples 50

# Analyze results
python3 -c "
import json
with open('results/detailed_results.json') as f:
    results = json.load(f)
print('Experiment completed successfully!')
for model in results:
    print(f'Model: {model}')
    for dataset in results[model]:
        print(f'  Dataset: {dataset}')
        for method in results[model][dataset]:
            ece = results[model][dataset][method]['ece']
            print(f'    {method}: ECE = {ece:.4f}')
"
```

## Key Implementation Details

### PBA Uncertainty Formula
```python
def pba_uncertainty(logits, targets, beta=0.5):
    probs = softmax(logits)
    target_probs = probs.gather(-1, targets.unsqueeze(-1))
    perplexities = 1.0 / target_probs
    uncertainties = 1.0 - exp(-beta * perplexities)
    return uncertainties.mean()
```

### Adjacent Possible Size
```python
def adjacent_possible_size(probs, alpha=0.9):
    sorted_probs = torch.sort(probs, descending=True)[0]
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    threshold_idx = (cumsum >= alpha).float().argmax(dim=-1)
    return threshold_idx + 1
```

### Calibration Evaluation
```python
def expected_calibration_error(uncertainties, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        in_bin = (uncertainties > bin_boundaries[i]) & (uncertainties <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = 1 - uncertainties[in_bin].mean()
            bin_size = in_bin.mean()
            ece += bin_size * abs(bin_acc - bin_conf)
    return ece
```