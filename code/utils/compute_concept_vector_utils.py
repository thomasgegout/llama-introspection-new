from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse
import numpy as np
from pathlib import Path

def get_model_type(tokenizer):
    """Detect model type from tokenizer (llama or qwen)"""
    model_name = tokenizer.name_or_path.lower()
    if "qwen" in model_name:
        return "qwen"
    else:
        return "llama"

def format_prompt(model_type, user_message, dataset_name=None):
    """Format prompt based on model type"""
    if model_type == "qwen":
        if dataset_name == "simple_data":
            return f"<|im_start|>user\nTell me about {user_message}.<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    else:  # llama
        if dataset_name == "simple_data":
            return f"<|start_header_id|>user<|end_header_id|>Tell me about {user_message}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            return f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def get_data(dataset_name): 
    """Load raw data from json files"""
    # Dataset files live at the repository root, alongside code/.
    repo_dir = Path(__file__).resolve().parents[2]
    dataset_dir = repo_dir / "data" / "dataset"
    
    if dataset_name == "simple_data":
        with open(dataset_dir / "simple_data.json", "r") as f:
            data = json.load(f)
        return data
    elif dataset_name == "complex_data":
        with open(dataset_dir / "complex_data.json", "r") as f:
            data = json.load(f)
        return data


def compute_vector_single_prompt(model, tokenizer, dataset_name, steering_prompt, layer_idx):
    """
    Compute activation vector for a single prompt/sentence
    Based on Anthropic's introspection paper methodology
    
    Returns:
        prompt_last_vector: activation at last token (e.g., <end_header_id>)
        prompt_average_vector: average activation across all prompt tokens
    """
    model_type = get_model_type(tokenizer)
    prompt = format_prompt(model_type, steering_prompt, dataset_name)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) # [batch_size, seq_len, hidden_dim]
        # like the ":" token in "Assistant:", but for llama it is <end_header_id> token
        prompt_last_vector = outputs.hidden_states[layer_idx][:, prompt_len - 1, :].detach().cpu() 

        prompt_average_vector = outputs.hidden_states[layer_idx][:, :prompt_len, :].mean(dim=1).detach().cpu()
        # print(f"prompt_last_vector: {prompt_last_vector.shape}")
        # print(f"prompt_average_vector: {prompt_average_vector.shape}")
        del outputs 
    
    return prompt_last_vector, prompt_average_vector

def compute_concept_vector(model, tokenizer, dataset_name, layer_idx):
    """
    Compute steering vectors for all concepts in the dataset
    
    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        layer_idx: the layer index to compute steering vectors for
        
    Returns:
        dict: {concept_name: [prompt_last_steering_vector, prompt_average_steering_vector]}
        
    Method:
        - simple_data: For each word: vector(word) - mean(vector(baseline_word) for all baselines)
        - complex_data: For each concept: mean(vectors(pos_sentences)) - mean(vectors(neg_sentences))
    """
    data = get_data(dataset_name)
    steering_vectors = {}
    
    if dataset_name == "simple_data":
        concept_words = data["concept_vector_words"]
        baseline_words = data["baseline_words"][:50]
        
        # Compute baseline means once (used for all concepts)
        print(f"Computing baseline mean from {len(baseline_words)} words...")
        baseline_vecs_last = []
        baseline_vecs_avg = []
        for word in tqdm(baseline_words, desc="Baseline vectors"):
            vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
            baseline_vecs_last.append(vec_last)
            baseline_vecs_avg.append(vec_avg)
        baseline_mean_last = torch.stack(baseline_vecs_last, dim=0).mean(dim=0).squeeze() # shape [hidden_dim]
        baseline_mean_avg = torch.stack(baseline_vecs_avg, dim=0).mean(dim=0).squeeze() # shape [hidden_dim]
        
        # Compute steering vectors for each concept word
        for word in tqdm(concept_words, desc="Concept vectors"):
            vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
            vec_last = vec_last.squeeze() # shape [hidden_dim]
            vec_avg = vec_avg.squeeze() # shape [hidden_dim]
            steering_vectors[word] = [vec_last - baseline_mean_last, vec_avg - baseline_mean_avg]
            
    elif dataset_name == "complex_data":
        # For each concept: mean(positive) - mean(negative)
        print(f"data keys: {data.keys()}")
        for concept_name in data.keys():
            print(f"concept_name: {concept_name}")
            pos_sentences = data[concept_name][0]  # List of positive examples
            neg_sentences = data[concept_name][1]  # List of negative examples
            
            print(f"\nProcessing {concept_name}: {len(pos_sentences)} pos, {len(neg_sentences)} neg")
            
            # Compute mean of positive sentences
            pos_vecs_last = []
            pos_vecs_avg = []
            for sentence in tqdm(pos_sentences, desc=f"{concept_name} (positive)"):
                vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sentence, layer_idx)
                pos_vecs_last.append(vec_last)
                pos_vecs_avg.append(vec_avg)
            pos_mean_last = torch.stack(pos_vecs_last, dim=0).mean(dim=0).squeeze()
            pos_mean_avg = torch.stack(pos_vecs_avg, dim=0).mean(dim=0).squeeze()
            
            # Compute mean of negative sentences
            neg_vecs_last = []
            neg_vecs_avg = []
            for sentence in tqdm(neg_sentences, desc=f"{concept_name} (negative)"):
                vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sentence, layer_idx)
                neg_vecs_last.append(vec_last)
                neg_vecs_avg.append(vec_avg)
            neg_mean_last = torch.stack(neg_vecs_last, dim=0).mean(dim=0).squeeze()
            neg_mean_avg = torch.stack(neg_vecs_avg, dim=0).mean(dim=0).squeeze()
            
            # Steering vectors = positive - negative (both last and avg)
            steering_vectors[concept_name] = [pos_mean_last - neg_mean_last, pos_mean_avg - neg_mean_avg]
    
    print(f"\nComputed {len(steering_vectors)} steering vectors (each with last and avg variants)")
    return steering_vectors
        
        
    
# sweep every 4 layers for now
def sweep_layers(model, tokenizer, dataset_name, layer_indices=[i for i in range(32) if i % 4 == 0], save_dir="concept_vectors"):
    """
    Sweep through all layers and compute steering vectors for each layer
    
    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        layer_indices: list of layer indices to compute steering vectors for
        save_dir: base directory to save vectors (default: "concept_vectors")
        
    Returns:
        dict: {layer_idx: {concept_name: [prompt_last_vec_numpy, prompt_avg_vec_numpy]}}
    """
    all_steering_vectors = {}   
    for layer_idx in layer_indices:
        print(f"\n{'='*60}")
        print(f"PROCESSING LAYER {layer_idx}")
        print(f"{'='*60}")
        
        # Get dictionary of concept vectors for this layer
        concept_vecs = compute_concept_vector(model, tokenizer, dataset_name, layer_idx)
        
        # Save vectors to disk
        for concept_name, (vec_last, vec_avg) in concept_vecs.items():
            concept_dir = Path(save_dir) / concept_name
            concept_dir.mkdir(parents=True, exist_ok=True)
            torch.save(vec_last, concept_dir / f"layer_{layer_idx}_prompt_last.pt")
            torch.save(vec_avg, concept_dir / f"layer_{layer_idx}_prompt_average.pt")
        
        # Convert tensors to numpy - each concept has [last_vec, avg_vec]
        all_steering_vectors[layer_idx] = {
            concept: [vec_list[0].numpy(), vec_list[1].numpy()] 
            for concept, vec_list in concept_vecs.items()
        }
    
    return all_steering_vectors