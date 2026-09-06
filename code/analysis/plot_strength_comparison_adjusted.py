#!/usr/bin/env python3
"""
Generate adjusted accuracy plot for strength comparison experiment.
Averaged over all concepts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Plot adjusted strength-comparison accuracy.")
parser.add_argument(
    "--input",
    type=Path,
    default=Path("plots/strength_comparison_all_concepts_best.pt"),
    help="Saved strength-comparison result file",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("plots/strength_comparison_all_concepts_adjusted_accuracy.png"),
    help="Output PNG path",
)
args = parser.parse_args()

# Load the aggregated data (all concepts)
data = torch.load(args.input, weights_only=False)

layers = data['layers']
strength_pairs = data['strength_pairs']
concepts = data['concepts']
avg_adj_acc = data['avg_adj_acc']
num_trials = data['num_trials']

print('='*100)
print('GENERATING ADJUSTED ACCURACY PLOT FOR STRENGTH COMPARISON')
print('='*100)
print()
print(f'Layers: {layers}')
print(f'Strength pairs: {strength_pairs}')
print(f'Concepts ({len(concepts)}): {concepts}')
print(f'Trials per config: {num_trials}')
print()

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(strength_pairs)))
markers = ['o', 's', '^', 'D']

for idx, pair in enumerate(strength_pairs):
    accuracies = []
    std_errs = []
    
    for layer in layers:
        key = (layer, pair)
        if key in avg_adj_acc:
            accuracies.append(avg_adj_acc[key]['mean'])
            # Standard error of the mean
            std_errs.append(avg_adj_acc[key]['std'] / np.sqrt(avg_adj_acc[key]['n']))
        else:
            accuracies.append(0.5)
            std_errs.append(0)
    
    ax.errorbar(layers, accuracies, yerr=std_errs, 
                label=f"α₁={pair[0]}, α₂={pair[1]}", 
                color=colors[idx], marker=markers[idx % len(markers)],
                linewidth=2, markersize=8, capsize=5)

# Baseline
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label="Chance (50%)")

ax.set_xlabel("Injection Layer", fontsize=14)
ax.set_ylabel("Adjusted Accuracy", fontsize=14)
ax.set_title("Strength Comparison: Can Model Distinguish Injection Strengths?\n"
             f"(Averaged over {len(concepts)} concepts, adjusted for baseline logit difference)",
             fontsize=14, fontweight='bold')
ax.set_ylim(0.3, 0.9)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
save_path = args.output
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'✓ Saved adjusted accuracy plot to: {save_path}')
print()

# Also print summary table
print('ADJUSTED ACCURACY TABLE:')
print(f"{'Layer':<8} ", end='')
for pair in strength_pairs:
    print(f'({pair[0]},{pair[1]})', end='     ')
print()
print('-'*80)

for layer in layers:
    print(f'{layer:<8} ', end='')
    for pair in strength_pairs:
        key = (layer, pair)
        if key in avg_adj_acc:
            val = avg_adj_acc[key]['mean']
            print(f'{val:>6.1%}', end='     ')
        else:
            print('  N/A ', end='     ')
    print()

print()
print('='*100)
print('KEY FINDINGS:')
print('='*100)

# Find best layer for each pair
print('Best layers for each strength pair:')
for pair in strength_pairs:
    best_layer = None
    best_acc = 0
    for layer in layers:
        key = (layer, pair)
        if key in avg_adj_acc:
            acc = avg_adj_acc[key]['mean']
            if acc > best_acc:
                best_acc = acc
                best_layer = layer
    
    if best_layer is not None:
        print(f'  ({pair[0]}, {pair[1]}): Layer {best_layer} = {best_acc:.1%}')

print()
print('Overall best config:')
best_overall = (None, None, 0)
for pair in strength_pairs:
    for layer in layers:
        key = (layer, pair)
        if key in avg_adj_acc:
            acc = avg_adj_acc[key]['mean']
            if acc > best_overall[2]:
                best_overall = (layer, pair, acc)

if best_overall[0] is not None:
    print(f'  Layer {best_overall[0]}, ({best_overall[1][0]}, {best_overall[1][1]}): {best_overall[2]:.1%}')

print()
print('✓ DONE!')

