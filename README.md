# Perplexity-Based Adjacency for Uncertainty Quantification in LLMs

This repository contains experimental validation of the PBA (Perplexity-Based Adjacency) uncertainty quantification method for Large Language Models, as described in the paper "Perplexity-Based Adjacency for Uncertainty Quantification in Large Language Models".

## Overview

The PBA method measures uncertainty through perplexity, treating high-entropy distributions as large possibility spaces and low-entropy distributions as constrained spaces. This eliminates arbitrary distance thresholds while providing uncertainty estimates calibrated to the model's learned distribution.

## Experiments

We validate the method using:
- **Models**: GPT-2, Qwen 2.5 3B, Gemma 2 2B, SmolLM2
- **Datasets**: TruthfulQA, MMLU subsets
- **Metrics**: Expected Calibration Error (ECE), Brier Score, computational efficiency
- **Baselines**: Softmax confidence, entropy-based methods

## Project Structure

```
├── src/
│   ├── uncertainty_methods.py    # PBA and baseline uncertainty implementations
│   ├── models.py                # Model interfaces
│   ├── metrics.py               # Evaluation metrics
│   ├── datasets.py              # Data loading and preprocessing
│   └── experiments.py           # Experiment orchestration
├── data/                        # Dataset storage
├── results/                     # Experimental results
├── notebooks/                   # Analysis notebooks
└── requirements.txt            # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/experiments.py
```
