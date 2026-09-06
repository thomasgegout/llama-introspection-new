# Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs

This repository contains the official implementation and code for the **arXiv preprint** of our paper *"Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs"*.

We systematically investigate whether large language models can introspect—that is, accurately detect perturbations to their own internal states—using activation steering in Meta-Llama-3.1-8B-Instruct. Our key findings: (1) binary detection tasks conflate introspection with global logit shifts; (2) on tasks requiring differential sensitivity (sentence localization, strength comparison), we find robust evidence for partial introspection (up to 88% localization accuracy, 83% strength discrimination); (3) these capabilities are confined to early-layer injections and explained mechanistically by attention-based signal routing and residual stream recovery dynamics.

**Paper:** [arXiv preprint link to be added]

## Repository Structure

```
├── code/
│   ├── experiments/          # Core experiment implementations
│   │   ├── main.py           # Main runner (localization, detection experiments)
│   │   ├── strength_comparison.py    # Strength comparison experiment
│   │   ├── position_detection.py     # Binary detection experiment
│   │   ├── mechinterp.py             # Mechanistic interpretability experiments
│   │   └── mechinterp_patching.py    # Activation patching experiments
│   ├── utils/                # Utility functions
│   │   ├── all_prompts.py            # All prompts used in experiments
│   │   ├── compute_concept_vector_utils.py  # Vector computation
│   │   ├── save_vectors.py           # Vector saving utilities
│   │   ├── inject_concept_vector.py  # Injection utilities
│   │   └── api_utils.py              # API utilities (GPT judges)
│   └── analysis/             # Analysis scripts
│       ├── analyze_detection_control.py      # Detection vs control analysis
│       ├── compute_position_detection_accuracy.py  # Position detection analysis
│       ├── plot_strength_comparison_adjusted.py    # Strength comparison plots
│       └── analyze_final.py          # Final analysis scripts
├── data/
│   ├── dataset/              # Concept datasets
│   │   ├── simple_data.json  # Simple concepts (Dust, Satellites, etc.)
│   │   └── complex_data.json # Complex concepts (betrayal, recursion, etc.)
│   └── saved_vectors/        # Pre-computed steering vectors
│       └── llama/            # Vectors for Llama-3.1-8B-Instruct
├── results/
│   ├── csv/                  # Experiment results in CSV format
│   └── mechinterp/           # Mechanistic interpretability results (.pt)
├── figures/                  # All figures used in paper
├── latex/                    # LaTeX paragraph drafts
└── scripts/                  # Shell scripts for running experiments
```

## Model

All experiments use **Meta-Llama-3.1-8B-Instruct** (32 layers, hidden dim 4096).

## Key Experiments

### 1. Binary Detection (Section 4)
- **File**: `code/experiments/position_detection.py`
- **Results**: `results/csv/output_control_question.csv`
- **Figures**: `figures/detection_vs_control_*.png`

### 2. Sentence Localization (Section 5.2)
- **File**: `code/experiments/main.py` (localization mode)
- **Results**: `results/csv/output_localization_*.csv`
- **Figures**: `figures/success_rate_localization_*.png`

### 3. Strength Comparison (Section 5.1)
- **File**: `code/experiments/strength_comparison.py`
- **Results**: `results/strength_comparison_all_concepts_all_concepts_best.pt`
- **Figures**: `figures/strength_comparison_*.png`

### 4. Mechanistic Analysis (Section 6)
- **File**: `code/experiments/mechinterp.py`, `mechinterp_patching.py`
- **Results**: `results/mechinterp/`
- **Figures**: `figures/mechinterp_*.png`

## Steering Vectors

Pre-computed vectors are in `data/saved_vectors/llama_reproduced/`:
- Format: `{concept}_{layer}_{vec_type}.pt`
- Vector types: `avg` (average across tokens), `last` (final token)
- 10 concepts × 32 layers × 2 types = 640 vectors

## Running Experiments

### Requirements
```bash
pip install torch transformers numpy pandas matplotlib tqdm
```

### Example: Strength Comparison
```bash
python code/experiments/strength_comparison.py \
    --concepts all \
    --layers 0 2 4 6 8 10 \
    --pairs best \
    --num_trials 30
```

### Example: Localization
```bash
python code/experiments/main.py \
    --type localization \
    --layers 0 1 2 3 4 5 8 11 14 17 20 \
    --coeffs 2 5 8 11 14 17 20 \
    --num_sentences 10 \
    --num_trials 50
```

## Concepts

**Simple (concrete nouns):** Dust, Satellites, Trumpets, Origami, Illusions

**Complex (abstract, contrastive):** fibonacci_numbers, recursion, betrayal, appreciation, shutdown

## Citation

If you use this code or build on our work, please cite our arXiv preprint:

```bibtex
@article{hahami2026detecting,
  title={Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs},
  author={Hahami, Ely and Sinha, Ishaan and Jain, Lavik and Kaplan, Josh and Hahami, Jon},
  journal={arXiv preprint},
  year={2026}
}
```

