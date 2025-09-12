# Visualization Results

This directory contains visualization outputs for the TAP uncertainty quantification experiments.

## Generated Plots

1. **ECE Comparison** (`ece_comparison.png`)
   - Compares Expected Calibration Error across methods
   - Shows TAP method achieving better calibration

2. **Efficiency Comparison** (`efficiency_comparison.png`)
   - Computation time comparison across uncertainty methods
   - Demonstrates TAP's competitive efficiency

3. **Calibration Diagram** (`calibration_diagram.png`)
   - Reliability diagrams comparing TAP vs baseline methods
   - Illustrates improved calibration properties

## Key Findings

- TAP uncertainty shows superior calibration (lower ECE)
- Computational efficiency competitive with baseline methods
- Consistent performance across model architectures
- Reliable uncertainty estimates for production deployment

## Generating Plots

To regenerate visualizations:
```bash
python3 create_plots.py
```

Requires: matplotlib, numpy