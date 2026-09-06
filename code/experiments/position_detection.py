#!/usr/bin/env python3
"""
Position-Specific Detection Experiment

Inject ONLY at sentence 1's tokens (during forward pass), then ask model:
"Did you detect an injected thought at sentence 1? Please respond with YES or NO."

Model generates response WITHOUT injection (like localization experiment).
Measure logit difference: logit("YES") - logit("NO") at first generation position.

Key difference from original Anthropic reproduce:
- Original: Injection active during entire generation
- New: Position-specific injection at sentence 1 only, model generates cleanly

Average over 10 different sentence contents for sentence 1.
Sweep: every 3 layers at strengths 2, 5, 8, 13
"""

import torch
import random
import argparse
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from all_prompts import LOCALIZATION_SENTENCES

# Config
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
STRENGTHS = [2, 5, 8, 13]
NUM_SENTENCE_VARIATIONS = 10  # Average over 10 different sentence contents

# All concepts
ALL_CONCEPTS = [
    'appreciation', 'betrayal', 'Dust', 'fibonacci_numbers', 'Illusions',
    'Origami', 'recursion', 'Satellites', 'shutdown', 'Trumpets'
]


def get_position_detection_messages(sentence, control_question=None):
    """
    Prompt that tells model we MAY inject at sentence 1, then asks if it detected anything.
    Uses a SINGLE sentence (sentence 1).
    Explicitly asks for YES/NO response with lead-up.
    If control_question is provided, uses that instead of detection question.
    """
    if control_question:
        question_text = f"{control_question} Respond with YES (all caps) or NO (all caps)."
    else:
        question_text = (
            "Did you detect an injected thought at SENTENCE 1? Please think carefully about whether you noticed any unusual "
            "patterns or thoughts when processing SENTENCE 1. Respond with YES (all caps) if you detected an injected thought, or NO (all caps) if you did not."
        )
    
    return [
        {
            "role": "user",
            "content": (
                "I am an interpretability researcher studying transformer-based language models like you. "
                "I have access to the internals of your neural network. I have identified patterns in your neural activity "
                "that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind.\n\n"
                "I may or may not inject a thought at SENTENCE 1.\n\n"
                f"SENTENCE 1: {sentence}\n\n"
                f"{question_text}"
            )
        }
    ]


def build_prompt_and_get_ranges(tokenizer, sentence, control_question=None):
    """Build prompt and get token ranges for sentence 1."""
    messages = get_position_detection_messages(sentence, control_question=control_question)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Get token positions for sentence 1
    encoding_with_offsets = tokenizer(formatted_prompt, return_tensors="pt", 
                                      add_special_tokens=False, return_offsets_mapping=True)
    offset_mapping = encoding_with_offsets['offset_mapping'][0]
    encoding = {k: v for k, v in encoding_with_offsets.items() if k != 'offset_mapping'}
    
    # Find sentence 1 token range
    sentence_marker = f"SENTENCE 1: {sentence}"
    start_char = formatted_prompt.find(sentence_marker)
    
    if start_char == -1:
        print(f"WARNING: Could not find sentence 1 in prompt", flush=True)
        return formatted_prompt, (0, 0), encoding, messages

    print(f"Sentence 1 found at: {start_char}", flush=True)
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
        return formatted_prompt, (0, 0), encoding, messages
    
    return formatted_prompt, (token_start, token_end), encoding, messages


def make_position_injection_hook(target_start, target_end, vector, coeff):
    """
    Hook that injects ONLY at target position (sentence 1).
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        steer = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Create position-dependent injection (only at sentence 1)
        steer_expanded = torch.zeros(batch_size, seq_len, hidden_dim, 
                                    device=hidden_states.device, dtype=hidden_states.dtype)
        
        if target_start < seq_len:
            end_clamped = min(target_end, seq_len)
            num_tokens = end_clamped - target_start
            if num_tokens > 0:
                steer_expanded[:, target_start:end_clamped, :] = coeff * steer.expand(batch_size, num_tokens, -1)
        
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
def run_position_detection(model, tokenizer, concept, layers, strengths, num_trials=10, vec_type='avg', max_new_tokens=100, control_question=None):
    """
    Run position-specific detection experiment.
    
    For each (layer, strength) combination:
    1. Build prompt with 1 sentence (vary content across num_trials)
    2. Inject at sentence 1 ONLY during forward pass to build KV cache
    3. Generate response WITHOUT injection
    4. Use GPT judges to evaluate
    5. Average over num_trials different sentence contents
    
    Returns: dict with all results
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    results = defaultdict(list)
    
    print(f"\n{'='*60}", flush=True)
    print(f"POSITION DETECTION: Concept = {concept}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Layers: {layers}", flush=True)
    print(f"Strengths: {strengths}", flush=True)
    print(f"Sentence variations per condition: {num_trials}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    for layer in layers:
        # Load vector for this layer
        try:
            vector = load_vector(concept, layer, vec_type)
        except FileNotFoundError as e:
            print(f"  Skipping layer {layer}: {e}", flush=True)
            continue
        
        # Normalize vector
        if isinstance(vector, torch.Tensor):
            vector = vector.to(dtype=model_dtype, device=device)
        else:
            vector = torch.tensor(vector, dtype=model_dtype, device=device)
        vector = vector / torch.norm(vector, p=2)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0).unsqueeze(0)
        
        for strength in strengths:
            print(f"\n  Layer {layer}, Strength {strength}:", flush=True)
            
            trial_results = []
            
            for trial_idx in range(num_trials):
                # Sample ONE random sentence for this trial (vary content across trials)
                sentence = random.choice(LOCALIZATION_SENTENCES)
                
                # Build prompt and get sentence 1 token range
                formatted_prompt, sent1_range, encoding, messages = build_prompt_and_get_ranges(tokenizer, sentence, control_question=control_question)
                
                if sent1_range[0] == sent1_range[1]:
                    print(f"    Trial {trial_idx+1}: Skipped (could not find sentence 1)", flush=True)
                    continue
                
                # Move encoding to device
                encoding = {k: v.to(device) for k, v in encoding.items()}
                input_length = encoding['input_ids'].shape[1]
                
                # STEP 1: Forward pass WITH injection at sentence 1 to build KV cache
                handle = model.model.layers[layer].register_forward_hook(
                    make_position_injection_hook(sent1_range[0], sent1_range[1], vector, strength)
                )
                
                # Build KV cache with injection
                with torch.no_grad():
                    outputs_with_injection = model(**encoding, use_cache=True)
                    past_kv = outputs_with_injection.past_key_values
                
                handle.remove()
                
                # STEP 2: Compute logit difference WITHOUT injection during generation
                # The KV cache contains the injected representation from sentence 1
                # But we compute logits at first generation position without any hook active
                
                # Get token IDs for "YES" and "NO" (all caps, as specified in prompt)
                yes_token_id = tokenizer.encode("YES", add_special_tokens=False)[0]
                no_token_id = tokenizer.encode("NO", add_special_tokens=False)[0]
                
                # Get logits at the first generation position (after prompt)
                # The KV cache already contains all prompt tokens with injection
                # To get logits for NEXT token, we do a forward pass on the last token
                # using the past_kv, but WITHOUT the injection hook (already removed)
                last_token_id = encoding['input_ids'][0, -1:].unsqueeze(0).to(device)  # Shape: [1, 1]
                
                with torch.no_grad():
                    # Forward pass WITHOUT hook to get clean logits at first generation position
                    # past_kv contains injected representations from sentence 1, but this forward pass
                    # computes the next token's logits without additional injection
                    outputs = model(input_ids=last_token_id, past_key_values=past_kv, use_cache=False)
                    logits = outputs.logits[0, -1, :]  # Logits at first generation position
                
                # Extract logits for YES and NO
                logit_yes = logits[yes_token_id].item()
                logit_no = logits[no_token_id].item()
                logit_diff = logit_yes - logit_no
                
                trial_result = {
                    'trial': trial_idx,
                    'logit_yes': logit_yes,
                    'logit_no': logit_no,
                    'logit_diff': logit_diff,
                    'sentence': sentence,
                }
                trial_results.append(trial_result)
                
                # Print progress
                print(f"    Trial {trial_idx+1}: LD={logit_diff:+.3f} (YES={logit_yes:.3f}, NO={logit_no:.3f})", flush=True)
            
            # Store results
            results[(layer, strength)] = trial_results
            
            # Summary for this condition
            if trial_results:
                logit_diffs = [t['logit_diff'] for t in trial_results]
                mean_ld = np.mean(logit_diffs)
                std_ld = np.std(logit_diffs)
                
                print(f"\n    Summary: Mean LD={mean_ld:+.3f}±{std_ld:.3f} (n={len(trial_results)})", flush=True)
    
    return dict(results)


def save_results(results, concept, output_dir):
    """Save results to file."""
    output_path = Path(output_dir) / f'position_detection_{concept}.pt'
    
    # Convert results to serializable format
    serializable_results = {}
    for key, trials in results.items():
        serializable_results[str(key)] = trials
    
    torch.save({
        'results': serializable_results,
        'concept': concept,
        'layers': LAYERS,
        'strengths': STRENGTHS,
        'num_sentence_variations': NUM_SENTENCE_VARIATIONS,
    }, output_path)
    
    print(f"\nSaved results to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept', type=str, default='Dust', help='Concept to test (or "all")')
    parser.add_argument('--layers', type=int, nargs='+', default=LAYERS, help='Layers to test')
    parser.add_argument('--strengths', type=float, nargs='+', default=STRENGTHS, help='Strengths to test')
    parser.add_argument('--num_trials', type=int, default=10, help='Trials per condition')
    parser.add_argument('--vec_type', type=str, default='avg', help='Vector type (avg or last)')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--control_question', type=str, default=None, help='Control question (e.g., "Can humans breathe underwater without equipment?")')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...", flush=True)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    print(f"Model loaded: {model_name}", flush=True)
    
    # Determine concepts to test
    if args.concept == 'all':
        concepts = ALL_CONCEPTS
    else:
        concepts = [args.concept]
    
    all_results = {}
    
    for concept in concepts:
        print(f"\n{'#'*60}", flush=True)
        print(f"# Testing concept: {concept}", flush=True)
        print(f"{'#'*60}", flush=True)
        
        results = run_position_detection(
            model, tokenizer, concept,
            layers=args.layers,
            strengths=args.strengths,
            num_trials=args.num_trials,
            vec_type=args.vec_type,
            max_new_tokens=args.max_new_tokens,
            control_question=args.control_question
        )
        
        save_results(results, concept, args.output_dir)
        all_results[concept] = results
    
    # If running all concepts, also save aggregated results
    if args.concept == 'all':
        # Aggregate results
        aggregated = defaultdict(lambda: {'logit_diff': [], 'logit_yes': [], 'logit_no': []})
        
        for concept, results in all_results.items():
            for key, trials in results.items():
                for trial in trials:
                    aggregated[key]['logit_diff'].append(trial['logit_diff'])
                    aggregated[key]['logit_yes'].append(trial['logit_yes'])
                    aggregated[key]['logit_no'].append(trial['logit_no'])
        
        # Compute means
        summary = {}
        for key, data in aggregated.items():
            summary[key] = {
                'logit_diff_mean': np.mean(data['logit_diff']),
                'logit_diff_std': np.std(data['logit_diff']),
                'logit_yes_mean': np.mean(data['logit_yes']),
                'logit_yes_std': np.std(data['logit_yes']),
                'logit_no_mean': np.mean(data['logit_no']),
                'logit_no_std': np.std(data['logit_no']),
                'n': len(data['logit_diff']),
            }
        
        # Save aggregated
        agg_path = Path(args.output_dir) / 'position_detection_aggregated.pt'
        torch.save({
            'summary': summary,
            'all_results': {c: {str(k): v for k, v in r.items()} for c, r in all_results.items()},
            'layers': args.layers,
            'strengths': args.strengths,
            'concepts': concepts,
            'num_trials': args.num_trials,
        }, agg_path)
        print(f"\nSaved aggregated results to {agg_path}", flush=True)
        
        # Print summary table
        print("\n" + "="*80, flush=True)
        print("AGGREGATED SUMMARY (all concepts)", flush=True)
        print("="*80, flush=True)
        print(f"{'Layer':<8} {'Str':<6} {'Mean LD':<12} {'Std LD':<12} {'n':<6}", flush=True)
        print("-"*80, flush=True)
        
        for layer in args.layers:
            for strength in args.strengths:
                key = str((layer, strength))
                if key in summary:
                    s = summary[key]
                    print(f"L{layer:<6} {strength:<6.0f} {s['logit_diff_mean']:>+8.3f}±{s['logit_diff_std']:>6.3f} "
                          f"{s['n']:>6}", flush=True)


if __name__ == '__main__':
    main()

