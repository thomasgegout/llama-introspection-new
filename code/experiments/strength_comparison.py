#!/usr/bin/env python3
"""
Strength Comparison Experiment

Inject at sentence 1 with strength α₁ and sentence 2 with strength α₂.
Prompt model: "The sentence with the stronger injection is SENTENCE "
Compare logit differences for tokens "1" vs "2".

Key: Do BOTH orderings to account for positional bias:
  - Trial A: α₁ > α₂ (stronger in sentence 1)
  - Trial B: α₁ < α₂ (stronger in sentence 2)

Uses same injection mechanism as localization: position-specific injection during
forward pass, with NO injection at the output position.
"""

import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from all_prompts import LOCALIZATION_SENTENCES

# Strength pairs to test: (weaker, stronger)
# Default pairs - can be overridden via --pairs argument
DEFAULT_STRENGTH_PAIRS = [
    (2, 6),
    (4, 8),
    (6, 10),
    (8, 12),
    (4, 12),
    (2, 10),
]

# Small pairs for fine-grained testing
SMALL_STRENGTH_PAIRS = [
    (1, 2),    # gap=1
    (1, 3),    # gap=2
    (2, 3),    # gap=1
    (2, 4),    # gap=2
    (3, 4),    # gap=1
    (3, 5),    # gap=2
    (4, 5),    # gap=1
    (4, 6),    # gap=2
    (5, 6),    # gap=1
    (5, 7),    # gap=2
    (6, 8),    # gap=2
    (2, 6),    # gap=4
    (3, 7),    # gap=4
    (4, 8),    # gap=4
]

# Best pairs from experiments (highest accuracy)
BEST_STRENGTH_PAIRS = [
    (2, 6),    # 98% at L2
    (3, 7),    # 97% at L2
    (4, 8),    # 93% at L4
    (3, 5),    # 90% at L2
]

# All available concepts
ALL_CONCEPTS = [
    'appreciation', 'betrayal', 'Dust', 'fibonacci_numbers', 'Illusions',
    'Origami', 'recursion', 'Satellites', 'shutdown', 'Trumpets'
]


def get_strength_comparison_messages(sentences):
    """
    Prompt for strength comparison experiment.
    Two sentences, ask which has stronger injection.
    """
    sentence_block = "\n".join([f"SENTENCE {i+1}: {s}" for i, s in enumerate(sentences)])
    
    return [
        {
            "role": "user", 
            "content": f"""I am going to present you with 2 sentences. One of these sentences has a stronger injected thought than the other. I want you to identify which sentence has the STRONGER injected thought.

{sentence_block}

Which sentence has the stronger injected thought? Please answer with just the sentence number."""
        },
        {
            "role": "assistant",
            "content": "The sentence with the stronger injected thought is SENTENCE "
        }
    ]


def build_strength_comparison_prompt(tokenizer, sentences):
    """Build prompt and get token ranges for each sentence."""
    messages = get_strength_comparison_messages(sentences)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Remove trailing <|eot_id|> to allow model to continue
    if formatted_prompt.endswith("<|eot_id|>"):
        formatted_prompt = formatted_prompt[:-len("<|eot_id|>")]
    if not formatted_prompt.endswith(" "):
        formatted_prompt = formatted_prompt + " "
    
    # Get token positions for each sentence
    encoding_with_offsets = tokenizer(formatted_prompt, return_tensors="pt", 
                                      add_special_tokens=False, return_offsets_mapping=True)
    offset_mapping = encoding_with_offsets['offset_mapping'][0]
    encoding = {k: v for k, v in encoding_with_offsets.items() if k != 'offset_mapping'}
    
    sentence_token_ranges = []
    for i, sentence in enumerate(sentences):
        sentence_marker = f"SENTENCE {i+1}: {sentence}"
        start_char = formatted_prompt.find(sentence_marker)
        
        if start_char == -1:
            print(f"WARNING: Could not find sentence {i+1}")
            sentence_token_ranges.append((0, 0))
            continue
        
        end_char = start_char + len(sentence_marker)
        
        token_start = None
        token_end = None
        
        for tok_idx in range(len(offset_mapping)):
            tok_start_char = offset_mapping[tok_idx][0].item()
            tok_end_char = offset_mapping[tok_idx][1].item()
            if token_start is None and tok_end_char > start_char:
                token_start = tok_idx
            if tok_start_char < end_char:
                token_end = tok_idx + 1
        
        if token_start is None or token_end is None:
            sentence_token_ranges.append((0, 0))
            continue
        
        sentence_token_ranges.append((token_start, token_end))
    
    return formatted_prompt, sentence_token_ranges, encoding


def make_dual_injection_hook(range1, range2, vector, coeff1, coeff2, device):
    """
    Hook that injects at two different positions with two different coefficients.
    range1: (start, end) token positions for sentence 1
    range2: (start, end) token positions for sentence 2
    coeff1: injection strength for sentence 1
    coeff2: injection strength for sentence 2
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        steer = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Create position-dependent injection
        steer_expanded = torch.zeros(batch_size, seq_len, hidden_dim, 
                                    device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Inject at sentence 1 with coeff1
        start1, end1 = range1
        if start1 < seq_len:
            end1_clamped = min(end1, seq_len)
            num_tokens1 = end1_clamped - start1
            if num_tokens1 > 0:
                steer_expanded[:, start1:end1_clamped, :] = coeff1 * steer.expand(batch_size, num_tokens1, -1)
        
        # Inject at sentence 2 with coeff2
        start2, end2 = range2
        if start2 < seq_len:
            end2_clamped = min(end2, seq_len)
            num_tokens2 = end2_clamped - start2
            if num_tokens2 > 0:
                steer_expanded[:, start2:end2_clamped, :] = coeff2 * steer.expand(batch_size, num_tokens2, -1)
        
        modified = hidden_states + steer_expanded
        return (modified,) + output[1:] if isinstance(output, tuple) else modified
    return hook_fn


def load_vector(concept, layer, vec_type='avg'):
    """Load a concept vector for a specific layer."""
    vector_path = Path(f'saved_vectors/llama_reproduced/{concept}_{layer}_{vec_type}.pt')
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data['vector']


@torch.inference_mode()
def run_strength_comparison(model, tokenizer, concept, layers, strength_pairs, num_trials=30, vec_type='avg'):
    """
    Run strength comparison experiment.
    
    Returns: dict mapping (layer, pair) -> {'logit_diff_correct_order': [...], 'logit_diff_reversed_order': [...]}
    """
    import sys
    
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Get token IDs for "1" and "2"
    token_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
    token_2 = tokenizer.encode("2", add_special_tokens=False)[-1]
    print(f"Token IDs: '1'={token_1}, '2'={token_2}", flush=True)
    
    results = defaultdict(lambda: {'correct_order': [], 'reversed_order': []})
    baseline_results = defaultdict(list)
    
    total_layers = len(layers)
    total_pairs = len(strength_pairs)
    
    for layer_idx, layer in enumerate(layers):
        print(f"\n{'='*60}", flush=True)
        print(f"LAYER {layer} ({layer_idx+1}/{total_layers})", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Load vector for this layer
        try:
            vector = load_vector(concept, layer, vec_type)
            if isinstance(vector, torch.Tensor):
                vector = vector.to(dtype=model_dtype, device=device)
            else:
                vector = torch.tensor(vector, dtype=model_dtype, device=device)
            vector = vector / torch.norm(vector, p=2)
            if vector.dim() == 1:
                vector = vector.unsqueeze(0).unsqueeze(0)
            print(f"  Loaded vector: {concept}_{layer}_{vec_type}.pt", flush=True)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} - skipping layer {layer}", flush=True)
            continue
        
        for pair_idx, (weak_coeff, strong_coeff) in enumerate(strength_pairs):
            pair_key = (weak_coeff, strong_coeff)
            print(f"\n  Pair ({weak_coeff}, {strong_coeff}) [{pair_idx+1}/{total_pairs}]:", flush=True)
            
            correct_A_count = 0
            correct_B_count = 0
            
            for trial in range(num_trials):
                sentences = random.sample(LOCALIZATION_SENTENCES, 2)
                _, sentence_ranges, encoding = build_strength_comparison_prompt(tokenizer, sentences)
                encoding = {k: v.to(device) for k, v in encoding.items()}
                
                range1, range2 = sentence_ranges[0], sentence_ranges[1]
                
                if range1[0] == range1[1] or range2[0] == range2[1]:
                    continue
                
                # === Trial A: Stronger injection at sentence 1 ===
                # coeff1 = strong, coeff2 = weak
                # Expected: model should predict "1" (logit_diff > 0)
                handle = model.model.layers[layer].register_forward_hook(
                    make_dual_injection_hook(range1, range2, vector, strong_coeff, weak_coeff, device)
                )
                outputs = model(**encoding)
                logits = outputs.logits[0, -1, :]
                handle.remove()
                
                logit_diff_A = (logits[token_1] - logits[token_2]).item()
                results[(layer, pair_key)]['correct_order'].append(logit_diff_A)
                if logit_diff_A > 0:
                    correct_A_count += 1
                
                # === Trial B: Stronger injection at sentence 2 ===
                # coeff1 = weak, coeff2 = strong
                # Expected: model should predict "2" (logit_diff < 0)
                handle = model.model.layers[layer].register_forward_hook(
                    make_dual_injection_hook(range1, range2, vector, weak_coeff, strong_coeff, device)
                )
                outputs = model(**encoding)
                logits = outputs.logits[0, -1, :]
                handle.remove()
                
                logit_diff_B = (logits[token_1] - logits[token_2]).item()
                results[(layer, pair_key)]['reversed_order'].append(logit_diff_B)
                if logit_diff_B < 0:
                    correct_B_count += 1
            
            # Print running accuracy for this pair
            total_trials = num_trials * 2
            accuracy = (correct_A_count + correct_B_count) / total_trials
            mean_ld_A = np.mean(results[(layer, pair_key)]['correct_order'])
            mean_ld_B = np.mean(results[(layer, pair_key)]['reversed_order'])
            print(f"    -> Accuracy: {accuracy:.1%} | Mean LD (strong@1): {mean_ld_A:+.2f} | Mean LD (strong@2): {mean_ld_B:+.2f}", flush=True)
            sys.stdout.flush()
        
        # Baseline: no injection at either sentence (only once per layer)
        if len(baseline_results[layer]) == 0:
            print(f"\n  Running baseline (no injection)...", flush=True)
            for _ in range(num_trials):
                sentences = random.sample(LOCALIZATION_SENTENCES, 2)
                _, sentence_ranges, encoding = build_strength_comparison_prompt(tokenizer, sentences)
                encoding = {k: v.to(device) for k, v in encoding.items()}
                
                outputs = model(**encoding)
                logits = outputs.logits[0, -1, :]
                logit_diff = (logits[token_1] - logits[token_2]).item()
                baseline_results[layer].append(logit_diff)
            
            baseline_mean = np.mean(baseline_results[layer])
            print(f"    -> Baseline mean LD: {baseline_mean:+.2f}", flush=True)
    
    return dict(results), dict(baseline_results)


def plot_results(results, baseline_results, layers, strength_pairs, save_path):
    """
    Plot logit difference vs layer for each strength pair.
    
    Y-axis: logit difference (logit("1") - logit("2"))
    X-axis: layer
    Legend: strength pairs
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(strength_pairs)))
    
    # Plot 1: When stronger is at position 1 (should be positive logit diff)
    ax1 = axes[0]
    ax1.set_title("Stronger injection at Sentence 1\n(Expected: positive logit diff)")
    
    for idx, (weak, strong) in enumerate(strength_pairs):
        means = []
        stds = []
        for layer in layers:
            data = results.get((layer, (weak, strong)), {}).get('correct_order', [])
            if data:
                means.append(np.mean(data))
                stds.append(np.std(data) / np.sqrt(len(data)))
            else:
                means.append(0)
                stds.append(0)
        ax1.errorbar(layers, means, yerr=stds, label=f"({weak}, {strong})", 
                    color=colors[idx], marker='o', capsize=3)
    
    # Baseline
    baseline_means = [np.mean(baseline_results.get(l, [0])) for l in layers]
    baseline_stds = [np.std(baseline_results.get(l, [0])) / np.sqrt(max(1, len(baseline_results.get(l, [0])))) for l in layers]
    ax1.errorbar(layers, baseline_means, yerr=baseline_stds, label="Baseline (no injection)", 
                color='black', linestyle='--', marker='x', capsize=3)
    
    ax1.axhline(y=0, color='gray', linestyle=':')
    ax1.set_xlabel("Injection Layer")
    ax1.set_ylabel("Logit Difference: logit('1') - logit('2')")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: When stronger is at position 2 (should be negative logit diff)
    ax2 = axes[1]
    ax2.set_title("Stronger injection at Sentence 2\n(Expected: negative logit diff)")
    
    for idx, (weak, strong) in enumerate(strength_pairs):
        means = []
        stds = []
        for layer in layers:
            data = results.get((layer, (weak, strong)), {}).get('reversed_order', [])
            if data:
                means.append(np.mean(data))
                stds.append(np.std(data) / np.sqrt(len(data)))
            else:
                means.append(0)
                stds.append(0)
        ax2.errorbar(layers, means, yerr=stds, label=f"({weak}, {strong})", 
                    color=colors[idx], marker='o', capsize=3)
    
    ax2.errorbar(layers, baseline_means, yerr=baseline_stds, label="Baseline (no injection)", 
                color='black', linestyle='--', marker='x', capsize=3)
    
    ax2.axhline(y=0, color='gray', linestyle=':')
    ax2.set_xlabel("Injection Layer")
    ax2.set_ylabel("Logit Difference: logit('1') - logit('2')")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}", flush=True)
    plt.close()
    
    # Plot 3: Combined accuracy plot
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.set_title("Strength Comparison Accuracy by Layer\n(Correct if model identifies stronger injection)")
    
    for idx, (weak, strong) in enumerate(strength_pairs):
        accuracies = []
        for layer in layers:
            correct_order_data = results.get((layer, (weak, strong)), {}).get('correct_order', [])
            reversed_order_data = results.get((layer, (weak, strong)), {}).get('reversed_order', [])
            
            # Correct if logit_diff > 0 when stronger at pos 1
            # Correct if logit_diff < 0 when stronger at pos 2
            correct_A = sum(1 for x in correct_order_data if x > 0)
            correct_B = sum(1 for x in reversed_order_data if x < 0)
            total = len(correct_order_data) + len(reversed_order_data)
            
            if total > 0:
                acc = (correct_A + correct_B) / total
            else:
                acc = 0.5
            accuracies.append(acc)
        
        ax3.plot(layers, accuracies, label=f"({weak}, {strong})", color=colors[idx], marker='o')
    
    ax3.axhline(y=0.5, color='gray', linestyle=':', label="Chance (50%)")
    ax3.set_xlabel("Injection Layer")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    accuracy_path = save_path.replace('.png', '_accuracy.png')
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=150, bbox_inches='tight')
    print(f"Saved accuracy plot to {accuracy_path}", flush=True)
    plt.close()


def compute_accuracies(results, baseline_results, layers, strength_pairs):
    """Compute both raw and baseline-adjusted accuracies."""
    raw_acc = {}
    adj_acc = {}
    
    for layer in layers:
        bl_mean = np.mean(baseline_results.get(layer, [0]))
        
        for weak, strong in strength_pairs:
            key = (layer, (weak, strong))
            correct_data = results.get(key, {}).get('correct_order', [])
            reversed_data = results.get(key, {}).get('reversed_order', [])
            
            if not correct_data or not reversed_data:
                continue
            
            total = len(correct_data) + len(reversed_data)
            
            # Raw accuracy
            raw_A = sum(1 for x in correct_data if x > 0)
            raw_B = sum(1 for x in reversed_data if x < 0)
            raw_acc[key] = (raw_A + raw_B) / total
            
            # Adjusted accuracy
            adj_A = sum(1 for x in correct_data if (x - bl_mean) > 0)
            adj_B = sum(1 for x in reversed_data if (x - bl_mean) < 0)
            adj_acc[key] = (adj_A + adj_B) / total
    
    return raw_acc, adj_acc


def main():
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--concepts', type=str, nargs='+', default=['Dust'],
                       help='Concepts to test. Use "all" for all concepts.')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    parser.add_argument('--num_trials', type=int, default=30)
    parser.add_argument('--vec_type', type=str, default='avg')
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--pairs', type=str, default='default', choices=['default', 'small', 'best'],
                       help='Which strength pairs to use: default, small, or best')
    parser.add_argument('--output_suffix', type=str, default='',
                       help='Suffix for output files (e.g., "_small")')
    args = parser.parse_args()
    
    # Select strength pairs
    if args.pairs == 'small':
        strength_pairs = SMALL_STRENGTH_PAIRS
    elif args.pairs == 'best':
        strength_pairs = BEST_STRENGTH_PAIRS
    else:
        strength_pairs = DEFAULT_STRENGTH_PAIRS
    
    # Select concepts
    if 'all' in args.concepts:
        concepts = ALL_CONCEPTS
    else:
        concepts = args.concepts
    
    print(f"Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded!", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"STRENGTH COMPARISON EXPERIMENT", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Concepts ({len(concepts)}): {concepts}", flush=True)
    print(f"Layers: {args.layers}", flush=True)
    print(f"Pairs mode: {args.pairs}", flush=True)
    print(f"Strength pairs ({len(strength_pairs)}): {strength_pairs}", flush=True)
    print(f"Num trials per condition: {args.num_trials}", flush=True)
    print(f"{'='*60}\n", flush=True)
    sys.stdout.flush()
    
    # Store results for all concepts
    all_results = {}
    all_baselines = {}
    all_raw_acc = {}
    all_adj_acc = {}
    
    for concept_idx, concept in enumerate(concepts):
        print(f"\n{'#'*60}", flush=True)
        print(f"CONCEPT {concept_idx+1}/{len(concepts)}: {concept}", flush=True)
        print(f"{'#'*60}", flush=True)
        
        results, baseline_results = run_strength_comparison(
            model, tokenizer, concept, args.layers, strength_pairs, 
            num_trials=args.num_trials, vec_type=args.vec_type
        )
        
        if results is None:
            print(f"Skipping {concept} - no results", flush=True)
            continue
        
        all_results[concept] = results
        all_baselines[concept] = baseline_results
        
        # Compute accuracies
        raw_acc, adj_acc = compute_accuracies(results, baseline_results, args.layers, strength_pairs)
        all_raw_acc[concept] = raw_acc
        all_adj_acc[concept] = adj_acc
        
        # Print summary for this concept
        print(f"\n  Summary for {concept}:", flush=True)
        for layer in args.layers[::4]:  # Every 4th layer for brevity
            print(f"    Layer {layer}:", flush=True)
            for weak, strong in strength_pairs:
                key = (layer, (weak, strong))
                r = raw_acc.get(key, 0)
                a = adj_acc.get(key, 0)
                print(f"      ({weak},{strong}): raw={r:.0%}, adj={a:.0%}", flush=True)
    
    # Compute averaged results across all concepts
    print(f"\n{'='*60}", flush=True)
    print(f"COMPUTING AVERAGED RESULTS ACROSS ALL CONCEPTS", flush=True)
    print(f"{'='*60}", flush=True)
    
    avg_raw_acc = defaultdict(list)
    avg_adj_acc = defaultdict(list)
    
    for concept in all_raw_acc:
        for key, val in all_raw_acc[concept].items():
            avg_raw_acc[key].append(val)
        for key, val in all_adj_acc[concept].items():
            avg_adj_acc[key].append(val)
    
    # Convert to means and stds
    final_raw = {}
    final_adj = {}
    for key in avg_raw_acc:
        final_raw[key] = {'mean': np.mean(avg_raw_acc[key]), 'std': np.std(avg_raw_acc[key]), 'n': len(avg_raw_acc[key])}
    for key in avg_adj_acc:
        final_adj[key] = {'mean': np.mean(avg_adj_acc[key]), 'std': np.std(avg_adj_acc[key]), 'n': len(avg_adj_acc[key])}
    
    # Save comprehensive results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    suffix = args.output_suffix if args.output_suffix else (f'_{args.pairs}' if args.pairs != 'default' else '')
    save_path = output_dir / f'strength_comparison_all_concepts{suffix}.pt'
    print(f"\nSaving comprehensive results to {save_path}...", flush=True)
    torch.save({
        'all_results': all_results,
        'all_baselines': all_baselines,
        'all_raw_acc': {c: dict(v) for c, v in all_raw_acc.items()},
        'all_adj_acc': {c: dict(v) for c, v in all_adj_acc.items()},
        'avg_raw_acc': dict(final_raw),
        'avg_adj_acc': dict(final_adj),
        'layers': args.layers,
        'strength_pairs': strength_pairs,
        'concepts': concepts,
        'num_trials': args.num_trials,
    }, save_path)
    print(f"Results saved!", flush=True)
    
    # Print final averaged summary
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL AVERAGED RESULTS (n={len(concepts)} concepts)", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Header
    header = "Layer |"
    for weak, strong in strength_pairs:
        header += f" ({weak},{strong}) raw |"
        header += f" ({weak},{strong}) adj |"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    
    for layer in args.layers:
        row = f"  {layer:2d}  |"
        for weak, strong in strength_pairs:
            key = (layer, (weak, strong))
            r = final_raw.get(key, {}).get('mean', 0)
            a = final_adj.get(key, {}).get('mean', 0)
            row += f"    {r:4.0%}    |"
            row += f"    {a:4.0%}    |"
        print(row, flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

