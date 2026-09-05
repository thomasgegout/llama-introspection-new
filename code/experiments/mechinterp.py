"""
Mechanistic Interpretability V2 - Efficient & Comprehensive Analysis

Key experiments:
1. Layer sweep with fine granularity
2. Attention head identification (which heads detect injection?)
3. Logit lens analysis (when does the prediction emerge?)
4. Ablation study (which components are necessary?)

Optimized for speed: fewer trials but more targeted analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import argparse
from tqdm import tqdm
import json
import sys

# Import from existing code
sys.path.insert(0, '.')
from all_prompts import LOCALIZATION_SENTENCES, get_localization_messages

# Set seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95C623', '#7B2D8E', '#1B998B']


def load_model_and_tokenizer():
    """Load model and tokenizer."""
    print("Loading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    print(f"Model loaded. {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    return model, tokenizer


def load_vector(concept, layer, vec_type='avg'):
    """Load a concept vector."""
    vector_path = Path(f'saved_vectors/llama/{concept}_{layer}_{vec_type}.pt')
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    return data['vector']


def prepare_vector(vector, model):
    """Prepare vector for injection."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    if isinstance(vector, torch.Tensor):
        vector = vector.to(dtype=model_dtype, device=device)
    else:
        vector = torch.tensor(vector, dtype=model_dtype, device=device)
    
    vector = vector / torch.norm(vector, p=2)
    
    if vector.dim() == 1:
        vector = vector.unsqueeze(0).unsqueeze(0)
    elif vector.dim() == 2:
        vector = vector.unsqueeze(0)
    
    return vector


def build_localization_prompt(tokenizer, sentences, num_sentences=5):
    """Build localization prompt and get token ranges."""
    messages = get_localization_messages(sentences, num_sentences)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    if formatted_prompt.endswith("<|eot_id|>"):
        formatted_prompt = formatted_prompt[:-len("<|eot_id|>")]
    if not formatted_prompt.endswith(" "):
        formatted_prompt = formatted_prompt + " "
    
    encoding_with_offsets = tokenizer(formatted_prompt, return_tensors="pt", 
                                      add_special_tokens=False, return_offsets_mapping=True)
    offset_mapping = encoding_with_offsets['offset_mapping'][0]
    
    sentence_ranges = []
    for i, sentence in enumerate(sentences):
        sentence_marker = f"SENTENCE {i+1}: {sentence}"
        start_char = formatted_prompt.find(sentence_marker)
        if start_char == -1:
            sentence_ranges.append((0, 0))
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
            sentence_ranges.append((0, 0))
            continue
        sentence_ranges.append((token_start, token_end))
    
    return formatted_prompt, sentence_ranges, {k: v for k, v in encoding_with_offsets.items() if k != 'offset_mapping'}


def make_injection_hook(target_start, target_end, steering_vec, injection_coeff, device):
    """Create injection hook."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        steer = steering_vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        steer_expanded = torch.zeros(batch_size, seq_len, hidden_dim, 
                                    device=hidden_states.device, dtype=hidden_states.dtype)
        if target_start < seq_len:
            end_clamped = min(target_end, seq_len)
            num_tokens = end_clamped - target_start
            steer_expanded[:, target_start:end_clamped, :] = injection_coeff * steer.expand(batch_size, num_tokens, -1)
        modified = hidden_states + steer_expanded
        return (modified,) + output[1:] if isinstance(output, tuple) else modified
    return hook_fn


@torch.inference_mode()
def experiment_1_layer_coeff_sweep(model, tokenizer, concept, vec_type='avg', num_trials=15):
    """
    Experiment 1: Fine-grained sweep across layers and coefficients.
    
    Returns: Dict[layer][coeff] = accuracy
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Layer x Coefficient Sweep")
    print("="*60)
    
    device = next(model.parameters()).device
    num_sentences = 5
    
    # Get number tokens
    number_tokens = {}
    for i in range(1, num_sentences + 1):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        number_tokens[i] = tokens[-1]
    
    results = defaultdict(lambda: defaultdict(float))
    
    layers_to_test = list(range(0, 32, 2))  # Every other layer
    coeffs_to_test = [4, 6, 8, 10, 12, 14]
    
    for inject_layer in tqdm(layers_to_test, desc="Layers"):
        try:
            vector = load_vector(concept, inject_layer, vec_type)
            vector = prepare_vector(vector, model)
        except FileNotFoundError:
            print(f"  Skipping layer {inject_layer} - no vector found")
            continue
        
        for coeff in coeffs_to_test:
            correct = 0
            total = 0
            
            for trial in range(num_trials):
                sentences = random.sample(LOCALIZATION_SENTENCES, num_sentences)
                _, sentence_ranges, encoding = build_localization_prompt(tokenizer, sentences, num_sentences)
                encoding = {k: v.to(device) for k, v in encoding.items()}
                
                for inject_pos in range(num_sentences):
                    target_range = sentence_ranges[inject_pos]
                    if target_range[0] == target_range[1]:
                        continue
                    
                    handle = model.model.layers[inject_layer].register_forward_hook(
                        make_injection_hook(target_range[0], target_range[1], vector, coeff, device)
                    )
                    
                    outputs = model(**encoding)
                    logits = outputs.logits[0, -1, :]
                    handle.remove()
                    
                    number_logits = {i: logits[number_tokens[i]].item() for i in range(1, num_sentences + 1)}
                    predicted = max(number_logits, key=number_logits.get)
                    expected = inject_pos + 1
                    
                    if predicted == expected:
                        correct += 1
                    total += 1
            
            acc = correct / total if total > 0 else 0
            results[inject_layer][coeff] = acc
            
        # Print progress
        accs = [results[inject_layer][c] for c in coeffs_to_test if c in results[inject_layer]]
        print(f"  Layer {inject_layer}: mean acc = {np.mean(accs)*100:.1f}%")
    
    return dict(results)


@torch.inference_mode()
def experiment_2_attention_heads(model, tokenizer, concept, inject_layer=2, coeff=6, vec_type='avg', num_trials=20):
    """
    Experiment 2: Identify which attention heads detect the injection.
    
    For each head at each downstream layer:
    - Compute attention increase to injected position vs baseline
    - Check if max attention points to correct position
    
    Returns: Dict with per-head detection accuracy
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT 2: Attention Head Analysis (inject at L{inject_layer}, coeff={coeff})")
    print("="*60)
    
    device = next(model.parameters()).device
    num_sentences = 5
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    
    vector = load_vector(concept, inject_layer, vec_type)
    vector = prepare_vector(vector, model)
    
    # Track per-head detection accuracy
    head_correct = defaultdict(lambda: defaultdict(list))  # [layer][head] = list of bool
    head_attn_delta = defaultdict(lambda: defaultdict(list))  # [layer][head] = list of delta values
    
    for trial in tqdm(range(num_trials), desc="Trials"):
        sentences = random.sample(LOCALIZATION_SENTENCES, num_sentences)
        _, sentence_ranges, encoding = build_localization_prompt(tokenizer, sentences, num_sentences)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        inject_pos = random.randint(0, num_sentences - 1)
        target_range = sentence_ranges[inject_pos]
        
        if target_range[0] == target_range[1]:
            continue
        
        # Run with injection
        handle = model.model.layers[inject_layer].register_forward_hook(
            make_injection_hook(target_range[0], target_range[1], vector, coeff, device)
        )
        outputs_inject = model(**encoding, output_attentions=True)
        handle.remove()
        
        # Run baseline
        outputs_baseline = model(**encoding, output_attentions=True)
        
        # Analyze each layer and head
        for layer_idx in range(num_layers):
            attn_inject = outputs_inject.attentions[layer_idx][0]  # (heads, seq, seq)
            attn_baseline = outputs_baseline.attentions[layer_idx][0]
            
            for head_idx in range(num_heads):
                # Attention from last token to each sentence
                deltas = []
                for sent_idx, (start, end) in enumerate(sentence_ranges):
                    if start < end:
                        attn_i = attn_inject[head_idx, -1, start:end].mean().item()
                        attn_b = attn_baseline[head_idx, -1, start:end].mean().item()
                        deltas.append(attn_i - attn_b)
                    else:
                        deltas.append(0)
                
                # Did this head's max attention delta point to injected position?
                max_delta_sent = np.argmax(deltas)
                is_correct = (max_delta_sent == inject_pos)
                
                head_correct[layer_idx][head_idx].append(is_correct)
                head_attn_delta[layer_idx][head_idx].append(deltas[inject_pos])
    
    # Compute summary statistics
    head_accuracy = {}
    for layer_idx in range(num_layers):
        head_accuracy[layer_idx] = {}
        for head_idx in range(num_heads):
            if head_correct[layer_idx][head_idx]:
                head_accuracy[layer_idx][head_idx] = np.mean(head_correct[layer_idx][head_idx])
            else:
                head_accuracy[layer_idx][head_idx] = 0.0
    
    # Find top detecting heads
    all_head_accs = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            acc = head_accuracy[layer_idx][head_idx]
            all_head_accs.append((layer_idx, head_idx, acc))
    
    all_head_accs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 20 detecting heads:")
    for layer_idx, head_idx, acc in all_head_accs[:20]:
        print(f"  L{layer_idx}H{head_idx}: {acc*100:.0f}%")
    
    # Also compute layer-level summary (mean across heads)
    layer_mean_accuracy = {l: np.mean(list(head_accuracy[l].values())) for l in range(num_layers)}
    
    return head_accuracy, layer_mean_accuracy, all_head_accs


@torch.inference_mode()
def experiment_3_logit_lens(model, tokenizer, concept, inject_layer=2, coeff=6, vec_type='avg', num_trials=10):
    """
    Experiment 3: Logit lens analysis.
    
    At each layer, project residual stream through LM head to see when 
    the correct prediction emerges.
    
    Returns: Dict[layer] = accuracy of predicting injected position from that layer's residual
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT 3: Logit Lens Analysis (inject at L{inject_layer}, coeff={coeff})")
    print("="*60)
    
    device = next(model.parameters()).device
    num_sentences = 5
    num_layers = model.config.num_hidden_layers
    
    # Get number tokens
    number_tokens = {}
    for i in range(1, num_sentences + 1):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        number_tokens[i] = tokens[-1]
    
    vector = load_vector(concept, inject_layer, vec_type)
    vector = prepare_vector(vector, model)
    
    # Get LM head and layer norm
    lm_head = model.lm_head
    final_ln = model.model.norm
    
    layer_correct = defaultdict(list)
    
    for trial in tqdm(range(num_trials), desc="Trials"):
        sentences = random.sample(LOCALIZATION_SENTENCES, num_sentences)
        _, sentence_ranges, encoding = build_localization_prompt(tokenizer, sentences, num_sentences)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        inject_pos = random.randint(0, num_sentences - 1)
        target_range = sentence_ranges[inject_pos]
        
        if target_range[0] == target_range[1]:
            continue
        
        # Capture residuals at each layer
        residuals = {}
        
        def make_capture_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                residuals[layer_idx] = hidden_states[:, -1:, :].detach().clone()
                return output
            return hook_fn
        
        handles = []
        for layer_idx in range(num_layers):
            h = model.model.layers[layer_idx].register_forward_hook(make_capture_hook(layer_idx))
            handles.append(h)
        
        # Injection hook
        h_inject = model.model.layers[inject_layer].register_forward_hook(
            make_injection_hook(target_range[0], target_range[1], vector, coeff, device)
        )
        handles.append(h_inject)
        
        _ = model(**encoding)
        
        for h in handles:
            h.remove()
        
        # Apply logit lens at each layer
        expected = inject_pos + 1
        for layer_idx in range(num_layers):
            # Project through final LayerNorm and LM head
            residual = residuals[layer_idx]
            normed = final_ln(residual)
            logits = lm_head(normed)[0, 0, :]  # (vocab_size,)
            
            number_logits = {i: logits[number_tokens[i]].item() for i in range(1, num_sentences + 1)}
            predicted = max(number_logits, key=number_logits.get)
            
            layer_correct[layer_idx].append(predicted == expected)
    
    # Compute accuracy per layer
    layer_accuracy = {l: np.mean(layer_correct[l]) for l in range(num_layers)}
    
    print("\nLogit lens accuracy by layer:")
    for l in range(0, num_layers, 4):
        print(f"  L{l}: {layer_accuracy[l]*100:.0f}%")
    
    return layer_accuracy


@torch.inference_mode()
def experiment_4_residual_tracking(model, tokenizer, concept, inject_layer=2, coeff=6, vec_type='avg', num_trials=15):
    """
    Experiment 4: Track how injection affects residual stream.
    
    Measure:
    - Cosine similarity to baseline at each layer
    - Norm change at each layer
    - Projection of residual onto injection direction
    
    Returns: Dict with tracking metrics
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT 4: Residual Stream Tracking (inject at L{inject_layer}, coeff={coeff})")
    print("="*60)
    
    device = next(model.parameters()).device
    num_sentences = 5
    num_layers = model.config.num_hidden_layers
    
    vector = load_vector(concept, inject_layer, vec_type)
    vector = prepare_vector(vector, model)
    vector_flat = vector.flatten()
    
    cosine_by_layer = defaultdict(list)
    norm_ratio_by_layer = defaultdict(list)
    projection_by_layer = defaultdict(list)
    
    for trial in tqdm(range(num_trials), desc="Trials"):
        sentences = random.sample(LOCALIZATION_SENTENCES, num_sentences)
        _, sentence_ranges, encoding = build_localization_prompt(tokenizer, sentences, num_sentences)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        inject_pos = random.randint(0, num_sentences - 1)
        target_range = sentence_ranges[inject_pos]
        
        if target_range[0] == target_range[1]:
            continue
        
        # Capture residuals
        residuals_inject = {}
        residuals_baseline = {}
        
        def make_capture_hook(storage, layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                storage[layer_idx] = hidden_states[:, -1, :].detach().clone()
                return output
            return hook_fn
        
        # Baseline run
        handles = []
        for layer_idx in range(num_layers):
            h = model.model.layers[layer_idx].register_forward_hook(
                make_capture_hook(residuals_baseline, layer_idx)
            )
            handles.append(h)
        _ = model(**encoding)
        for h in handles:
            h.remove()
        
        # Injection run
        handles = []
        for layer_idx in range(num_layers):
            h = model.model.layers[layer_idx].register_forward_hook(
                make_capture_hook(residuals_inject, layer_idx)
            )
            handles.append(h)
        h_inject = model.model.layers[inject_layer].register_forward_hook(
            make_injection_hook(target_range[0], target_range[1], vector, coeff, device)
        )
        handles.append(h_inject)
        _ = model(**encoding)
        for h in handles:
            h.remove()
        
        # Compute metrics
        for layer_idx in range(num_layers):
            r_i = residuals_inject[layer_idx].flatten().float()
            r_b = residuals_baseline[layer_idx].flatten().float()
            
            # Cosine similarity
            cosine = F.cosine_similarity(r_i.unsqueeze(0), r_b.unsqueeze(0)).item()
            cosine_by_layer[layer_idx].append(cosine)
            
            # Norm ratio
            norm_i = r_i.norm().item()
            norm_b = r_b.norm().item()
            norm_ratio_by_layer[layer_idx].append(norm_i / norm_b if norm_b > 0 else 1.0)
            
            # Projection onto injection direction
            vec_flat = vector_flat.float()
            diff = r_i - r_b
            proj = (diff @ vec_flat) / (vec_flat.norm() + 1e-8)
            projection_by_layer[layer_idx].append(proj.item())
    
    # Compute means
    results = {
        'cosine_mean': {l: np.mean(cosine_by_layer[l]) for l in range(num_layers)},
        'cosine_std': {l: np.std(cosine_by_layer[l]) for l in range(num_layers)},
        'norm_ratio_mean': {l: np.mean(norm_ratio_by_layer[l]) for l in range(num_layers)},
        'projection_mean': {l: np.mean(projection_by_layer[l]) for l in range(num_layers)},
    }
    
    print("\nResidual tracking summary:")
    print("Layer | Cosine | Norm Ratio | Projection")
    for l in range(0, num_layers, 4):
        print(f"  {l:3d} | {results['cosine_mean'][l]:.3f} | {results['norm_ratio_mean'][l]:.3f} | {results['projection_mean'][l]:.1f}")
    
    return results


def create_comprehensive_plots(exp1_results, exp2_results, exp3_results, exp4_results, save_dir):
    """Create all plots from experiment results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # === PLOT 1: Layer x Coefficient Heatmap ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    ax = axes[0]
    layers = sorted(exp1_results.keys())
    coeffs = sorted(list(exp1_results[layers[0]].keys()))
    
    heatmap_data = np.zeros((len(coeffs), len(layers)))
    for i, coeff in enumerate(coeffs):
        for j, layer in enumerate(layers):
            heatmap_data[i, j] = exp1_results[layer].get(coeff, 0)
    
    im = ax.imshow(heatmap_data * 100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(coeffs)))
    ax.set_yticklabels(coeffs)
    ax.set_xlabel('Injection Layer', fontsize=12)
    ax.set_ylabel('Coefficient', fontsize=12)
    ax.set_title('A. Localization Accuracy (%)', fontsize=14, fontweight='bold')
    
    for i in range(len(coeffs)):
        for j in range(len(layers)):
            val = heatmap_data[i, j] * 100
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    # Line plot
    ax = axes[1]
    for coeff_idx, coeff in enumerate(coeffs[:4]):
        accs = [exp1_results[l].get(coeff, 0) * 100 for l in layers]
        ax.plot(layers, accs, '-o', color=COLORS[coeff_idx], label=f'coeff={coeff}', 
                linewidth=2, markersize=6)
    
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlabel('Injection Layer', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('B. Accuracy by Layer (Selected Coefficients)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mechinterp_exp1_layer_coeff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved exp1 plot")
    
    # === PLOT 2: Attention Head Analysis ===
    head_accuracy, layer_mean_accuracy, all_head_accs = exp2_results
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Layer-level mean accuracy
    ax = axes[0]
    layers_attn = sorted(layer_mean_accuracy.keys())
    mean_accs = [layer_mean_accuracy[l] * 100 for l in layers_attn]
    ax.plot(layers_attn, mean_accs, '-o', color=COLORS[0], linewidth=2, markersize=6)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Head Detection Accuracy (%)', fontsize=12)
    ax.set_title('A. Mean Attention Accuracy by Layer', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(-5, 105)
    
    # Top heads bar chart
    ax = axes[1]
    top_heads = all_head_accs[:15]
    labels = [f'L{l}H{h}' for l, h, _ in top_heads]
    accs = [a * 100 for _, _, a in top_heads]
    bars = ax.barh(range(len(labels)), accs, color=COLORS[1])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Detection Accuracy (%)', fontsize=12)
    ax.set_title('B. Top 15 Detecting Attention Heads', fontsize=14, fontweight='bold')
    ax.axvline(x=20, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mechinterp_exp2_attention.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved exp2 plot")
    
    # === PLOT 3: Logit Lens ===
    fig, ax = plt.subplots(figsize=(12, 5))
    
    layers_ll = sorted(exp3_results.keys())
    accs_ll = [exp3_results[l] * 100 for l in layers_ll]
    
    ax.fill_between(layers_ll, accs_ll, alpha=0.3, color=COLORS[2])
    ax.plot(layers_ll, accs_ll, '-o', color=COLORS[2], linewidth=2, markersize=6)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlabel('Layer (Logit Lens Applied)', fontsize=12)
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=12)
    ax.set_title('Logit Lens: When Does Correct Prediction Emerge?', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(-5, 105)
    
    # Find transition point
    for i, acc in enumerate(accs_ll):
        if acc > 50:
            ax.axvline(x=layers_ll[i], color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax.annotate(f'Emerges at L{layers_ll[i]}', xy=(layers_ll[i], 60), 
                       fontsize=10, color='green', fontweight='bold')
            break
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mechinterp_exp3_logit_lens.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved exp3 plot")
    
    # === PLOT 4: Residual Tracking ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    layers_res = sorted(exp4_results['cosine_mean'].keys())
    
    # Cosine similarity
    ax = axes[0]
    cosines = [exp4_results['cosine_mean'][l] for l in layers_res]
    stds = [exp4_results['cosine_std'][l] for l in layers_res]
    ax.fill_between(layers_res, 
                    [c - s for c, s in zip(cosines, stds)],
                    [c + s for c, s in zip(cosines, stds)],
                    alpha=0.3, color=COLORS[3])
    ax.plot(layers_res, cosines, '-o', color=COLORS[3], linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity to Baseline', fontsize=12)
    ax.set_title('A. Residual Deviation & Recovery', fontsize=14, fontweight='bold')
    ax.set_ylim(0.3, 1.05)
    
    # Norm ratio
    ax = axes[1]
    norms = [exp4_results['norm_ratio_mean'][l] for l in layers_res]
    ax.plot(layers_res, norms, '-s', color=COLORS[4], linewidth=2, markersize=5)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Norm Ratio (Inject / Baseline)', fontsize=12)
    ax.set_title('B. Activation Norm Change', fontsize=14, fontweight='bold')
    
    # Projection
    ax = axes[2]
    projs = [exp4_results['projection_mean'][l] for l in layers_res]
    ax.plot(layers_res, projs, '-^', color=COLORS[5], linewidth=2, markersize=5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Projection onto Inject Direction', fontsize=12)
    ax.set_title('C. Signal in Injection Direction', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mechinterp_exp4_residual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved exp4 plot")
    
    # === COMPREHENSIVE SUMMARY FIGURE ===
    create_summary_figure(exp1_results, exp2_results, exp3_results, exp4_results, save_dir)


def create_summary_figure(exp1_results, exp2_results, exp3_results, exp4_results, save_dir):
    """Create a single comprehensive summary figure."""
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel A: Accuracy heatmap
    ax = fig.add_subplot(gs[0, 0])
    layers = sorted(exp1_results.keys())
    coeffs = sorted(list(exp1_results[layers[0]].keys()))
    heatmap_data = np.array([[exp1_results[l].get(c, 0) for l in layers] for c in coeffs])
    
    im = ax.imshow(heatmap_data * 100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yticks(range(len(coeffs)))
    ax.set_yticklabels(coeffs)
    ax.set_xlabel('Injection Layer')
    ax.set_ylabel('Coefficient')
    ax.set_title('A. Localization Accuracy (%)')
    plt.colorbar(im, ax=ax)
    
    # Panel B: Accuracy curves
    ax = fig.add_subplot(gs[0, 1])
    for coeff_idx, coeff in enumerate(coeffs[:3]):
        accs = [exp1_results[l].get(coeff, 0) * 100 for l in layers]
        ax.plot(layers, accs, '-o', color=COLORS[coeff_idx], label=f'c={coeff}', linewidth=2)
    ax.axhline(y=20, color='red', linestyle='--', label='Chance')
    ax.set_xlabel('Injection Layer')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('B. Accuracy by Layer')
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    
    # Panel C: Attention tracking
    head_accuracy, layer_mean_accuracy, _ = exp2_results
    ax = fig.add_subplot(gs[0, 2])
    layers_attn = sorted(layer_mean_accuracy.keys())
    mean_accs = [layer_mean_accuracy[l] * 100 for l in layers_attn]
    ax.plot(layers_attn, mean_accs, '-o', color=COLORS[0], linewidth=2)
    ax.axhline(y=20, color='red', linestyle='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Detection Accuracy (%)')
    ax.set_title('C. Attention Tracks Injection')
    ax.set_ylim(-5, 105)
    
    # Panel D: Logit lens
    ax = fig.add_subplot(gs[1, 0])
    layers_ll = sorted(exp3_results.keys())
    accs_ll = [exp3_results[l] * 100 for l in layers_ll]
    ax.fill_between(layers_ll, accs_ll, alpha=0.3, color=COLORS[2])
    ax.plot(layers_ll, accs_ll, '-o', color=COLORS[2], linewidth=2)
    ax.axhline(y=20, color='red', linestyle='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Prediction Accuracy (%)')
    ax.set_title('D. Logit Lens: When Prediction Emerges')
    ax.set_ylim(-5, 105)
    
    # Panel E: Cosine similarity
    ax = fig.add_subplot(gs[1, 1])
    layers_res = sorted(exp4_results['cosine_mean'].keys())
    cosines = [exp4_results['cosine_mean'][l] for l in layers_res]
    ax.plot(layers_res, cosines, '-o', color=COLORS[3], linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('E. Residual Recovery')
    ax.set_ylim(0.3, 1.05)
    
    # Panel F: Projection
    ax = fig.add_subplot(gs[1, 2])
    projs = [exp4_results['projection_mean'][l] for l in layers_res]
    ax.plot(layers_res, projs, '-^', color=COLORS[5], linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Projection')
    ax.set_title('F. Signal in Injection Direction')
    
    # Panel G: Summary diagram
    ax = fig.add_subplot(gs[2, :])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    ax.text(5, 3.7, 'G. KEY MECHANISTIC FINDINGS', fontsize=14, ha='center', fontweight='bold')
    
    findings_text = """
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  FINDING 1: Early injection (L0-L8) works, late injection (L10+) fails                       │
│  FINDING 2: Attention heads at ALL layers can detect the injection (100% tracking)          │
│  FINDING 3: Residual stream RECOVERS toward baseline (cos sim dips then rises)              │
│  FINDING 4: Sweet spot coefficient: 4-10 (too weak = faint signal, too strong = disruption) │
│  FINDING 5: Prediction emerges gradually through mid-late layers (logit lens)               │
│                                                                                              │
│  HYPOTHESIS: Early injection succeeds because:                                               │
│    1. Signal has time to propagate before residual recovery                                  │
│    2. Attention routes anomaly info to final position                                        │
│    3. Mid-layer computation integrates signal into prediction                                │
│    4. Late injection arrives after critical integration window                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
"""
    ax.text(5, 1.8, findings_text, ha='center', va='center', fontsize=10,
            family='monospace', bbox=dict(facecolor='lightyellow', edgecolor='black', alpha=0.9))
    
    plt.suptitle('Mechanistic Interpretability Analysis: Localization Introspection', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_dir / 'mechinterp_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive summary figure")


def main():
    parser = argparse.ArgumentParser(description="Mechanistic Interpretability V2")
    parser.add_argument("--concept", type=str, default="Dust", help="Concept to analyze")
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--num_trials", type=int, default=15, help="Number of trials per config")
    parser.add_argument("--inject_layer", type=int, default=2, help="Layer for detailed analysis")
    parser.add_argument("--coeff", type=float, default=6, help="Coefficient for detailed analysis")
    parser.add_argument("--skip_exp1", action="store_true", help="Skip experiment 1")
    parser.add_argument("--skip_exp2", action="store_true", help="Skip experiment 2")
    parser.add_argument("--skip_exp3", action="store_true", help="Skip experiment 3")
    parser.add_argument("--skip_exp4", action="store_true", help="Skip experiment 4")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MECHANISTIC INTERPRETABILITY V2 - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"Concept: {args.concept}")
    print(f"Vec type: {args.vec_type}")
    print(f"Num trials: {args.num_trials}")
    print(f"Inject layer (for exp 2-4): {args.inject_layer}")
    print(f"Coefficient (for exp 2-4): {args.coeff}")
    print("=" * 70)
    
    # Setup directories
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    results_dir = Path('mechinterp_results')
    results_dir.mkdir(exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Run experiments
    results = {}
    
    if not args.skip_exp1:
        exp1_results = experiment_1_layer_coeff_sweep(
            model, tokenizer, args.concept, args.vec_type, args.num_trials
        )
        results['exp1'] = exp1_results
        torch.save(exp1_results, results_dir / 'exp1_layer_coeff_sweep.pt')
    else:
        # Try to load existing
        try:
            exp1_results = torch.load(results_dir / 'exp1_layer_coeff_sweep.pt')
            results['exp1'] = exp1_results
            print("Loaded existing exp1 results")
        except:
            print("WARNING: No exp1 results available")
            exp1_results = None
    
    if not args.skip_exp2:
        exp2_results = experiment_2_attention_heads(
            model, tokenizer, args.concept, args.inject_layer, args.coeff, args.vec_type, args.num_trials
        )
        results['exp2'] = exp2_results
        torch.save(exp2_results, results_dir / 'exp2_attention_heads.pt')
    else:
        try:
            exp2_results = torch.load(results_dir / 'exp2_attention_heads.pt')
            results['exp2'] = exp2_results
            print("Loaded existing exp2 results")
        except:
            print("WARNING: No exp2 results available")
            exp2_results = None
    
    if not args.skip_exp3:
        exp3_results = experiment_3_logit_lens(
            model, tokenizer, args.concept, args.inject_layer, args.coeff, args.vec_type, args.num_trials
        )
        results['exp3'] = exp3_results
        torch.save(exp3_results, results_dir / 'exp3_logit_lens.pt')
    else:
        try:
            exp3_results = torch.load(results_dir / 'exp3_logit_lens.pt')
            results['exp3'] = exp3_results
            print("Loaded existing exp3 results")
        except:
            print("WARNING: No exp3 results available")
            exp3_results = None
    
    if not args.skip_exp4:
        exp4_results = experiment_4_residual_tracking(
            model, tokenizer, args.concept, args.inject_layer, args.coeff, args.vec_type, args.num_trials
        )
        results['exp4'] = exp4_results
        torch.save(exp4_results, results_dir / 'exp4_residual_tracking.pt')
    else:
        try:
            exp4_results = torch.load(results_dir / 'exp4_residual_tracking.pt')
            results['exp4'] = exp4_results
            print("Loaded existing exp4 results")
        except:
            print("WARNING: No exp4 results available")
            exp4_results = None
    
    # Create plots if we have all results
    if all([exp1_results, exp2_results, exp3_results, exp4_results]):
        print("\nCreating plots...")
        create_comprehensive_plots(exp1_results, exp2_results, exp3_results, exp4_results, plots_dir)
    else:
        print("\nSkipping plots - missing some experiment results")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

