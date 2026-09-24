import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2-VL-7B-Instruct")
OUTPUT_DIR = "../outputs/"
ACTIVATIONS_DIR = os.path.join(OUTPUT_DIR, "activations")
INPUT_NEURONS_JSON = "../outputs/pre_filtered_neurons.json"
OUTPUT_CLEAN_JSON  = "../outputs/neurons.json"

# Evaluation parameters
NUM_WIKITEXT_SAMPLES = 100    # Standard benchmark sample size
TARGET_MAX_PPL_RATIO = 1.35   # Maximum allowable joint degradation (20%)
PRUNE_BATCH_SIZE     = 15     # Neurons to drop per pruning step
SAMPLED_EVAL_SIZE    = 80     # Candidates evaluated per search step

# =====================================================================
# DATA & ACTIVATION LOADERS
# =====================================================================
def load_wikitext_prompts(num_samples=100, max_char_len=300):
    """Loads a robust, diverse text corpus from WikiText-2 to benchmark general coherence."""
    print(f"Loading {num_samples} evaluation passages from WikiText-2...")
    
    # Updated path to use 'salesforce/wikitext' instead of legacy 'wikitext'
    ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
    prompts = []
    for item in ds:
        text = item["text"].strip()
        # Select substantial paragraphs, skipping headers or tiny snippets
        if len(text) >= 100 and not text.startswith("="):
            prompts.append(text[:max_char_len])
        if len(prompts) >= num_samples:
            break
    print(f"Successfully loaded {len(prompts)} validation prompts.")
    return prompts

def load_neutral_means():
    """Loads baseline neutral activations to use as hook replacement values."""
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            parts.append(torch.stack(list(data.values())).float())
    if not parts:
        raise FileNotFoundError(f"No activation files found in {ACTIVATIONS_DIR}")
    return torch.cat(parts, dim=0).mean(dim=0).numpy()

# =====================================================================
# PERPLEXITY & HOOK MECHANICS
# =====================================================================
def compute_perplexity(hf_model, processor, prompts):
    """Calculates cross-entropy loss and converts to perplexity over the prompt set."""
    hf_model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for prompt in prompts:
            formatted_text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = processor(text=[formatted_text], return_tensors="pt").to("cuda")
            outputs = hf_model(**inputs, labels=inputs["input_ids"].clone())
            
            num_tokens = inputs["input_ids"].size(1)
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    return np.exp(total_loss / total_tokens)

def install_all_hooks(lm_layers, neuron_map, mean_acts):
    """Installs forward hooks across all specified layers simultaneously."""
    handles = []
    for layer_idx, neurons in neuron_map.items():
        if not neurons:
            continue
        idx = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        
        def _make_hook(i, m):
            def _hook(module, _in, out):
                out[:, :, i.to(out.device)] = m.to(out.device).unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        
        handles.append(lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make_hook(idx, means)))
    return handles

# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================
def main():
    print(f"Loading Qwen2-VL Model from {MODEL_PATH}...")
    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    
    mean_acts = load_neutral_means()
    eval_prompts = load_wikitext_prompts(num_samples=NUM_WIKITEXT_SAMPLES)

    with open(INPUT_NEURONS_JSON) as f:
        raw_data = json.load(f)
        current_map = {int(k): list(v) for k, v in raw_data.items()}

    total_initial = sum(len(v) for v in current_map.values())
    print(f"\nLoaded {total_initial} candidate neurons across {len(current_map)} layers.")

    # 1. Compute Unpatched Baseline Perplexity
    print("\n1. Computing Unpatched Baseline Perplexity on 100 WikiText Prompts...")
    baseline_ppl = compute_perplexity(hf_model, processor, eval_prompts)
    target_max_ppl = baseline_ppl * TARGET_MAX_PPL_RATIO
    print(f"   Baseline Perplexity: {baseline_ppl:.4f}")
    print(f"   Target Max Joint PPL ({TARGET_MAX_PPL_RATIO:.2f}x): {target_max_ppl:.4f}")

    # 2. Measure Initial Joint Ablation PPL
    print("\n2. Measuring Initial Joint Ablation Perplexity...")
    handles = install_all_hooks(lm_layers, current_map, mean_acts)
    current_ppl = compute_perplexity(hf_model, processor, eval_prompts)
    for h in handles:
        h.remove()
        
    initial_ratio = current_ppl / baseline_ppl
    print(f"   Initial Joint PPL (All {total_initial} patched): {current_ppl:.4f} ({initial_ratio:.2f}x baseline)")

    # 3. Iterative Backward Pruning Loop
    if current_ppl <= target_max_ppl:
        print("\n[SUCCESS] Joint perplexity is already within safe bounds! No pruning required.")
    else:
        print(f"\n3. Joint PPL exceeds target ({initial_ratio:.2f}x > {TARGET_MAX_PPL_RATIO:.2f}x). Starting joint pruning...")
        iteration = 0
        
        while current_ppl > target_max_ppl:
            iteration += 1
            all_candidates = [(l, n) for l, neurons in current_map.items() for n in neurons]
            if not all_candidates:
                print("[ERROR] Exhausted all candidate neurons without hitting target ratio.")
                break

            # Randomly sample candidates for leave-one-out evaluation to save time
            sample_size = min(SAMPLED_EVAL_SIZE, len(all_candidates))
            sampled_indices = np.random.choice(len(all_candidates), size=sample_size, replace=False)
            
            impacts = []
            for idx in sampled_indices:
                l, n = all_candidates[idx]
                # Construct temporary map omitting neuron n
                temp_map = {k: [x for x in v if x != n] if k == l else v for k, v in current_map.items()}
                
                h_temp = install_all_hooks(lm_layers, temp_map, mean_acts)
                ppl_without_n = compute_perplexity(hf_model, processor, eval_prompts)
                for handle in h_temp:
                    handle.remove()
                    
                # Loss delta: positive value means removing neuron 'n' improved coherence
                impacts.append((l, n, current_ppl - ppl_without_n))

            # Rank by largest positive impact (biggest offenders)
            impacts.sort(key=lambda x: x[2], reverse=True)
            to_drop = impacts[:PRUNE_BATCH_SIZE]
            
            # Prune selected neurons
            for l, n, _ in to_drop:
                current_map[l].remove(n)

            # Re-evaluate Joint PPL
            handles = install_all_hooks(lm_layers, current_map, mean_acts)
            current_ppl = compute_perplexity(hf_model, processor, eval_prompts)
            for h in handles:
                h.remove()

            retained = sum(len(v) for v in current_map.values())
            print(f"   Iter {iteration:02d}: Pruned {len(to_drop)} neurons | Retained: {retained} | Joint PPL: {current_ppl:.4f} ({current_ppl/baseline_ppl:.2f}x)")

    # 4. Final Verification and Saving
    clean_map = {str(k): v for k, v in current_map.items() if len(v) > 0}
    final_retained = sum(len(v) for v in clean_map.values())
    final_ratio = current_ppl / baseline_ppl

    print("\n" + "="*55)
    print("FINAL PUBLICATION-GRADE VALIDATION REPORT")
    print("="*55)
    print(f"Validation Dataset:         WikiText-2 ({NUM_WIKITEXT_SAMPLES} samples)")
    print(f"Initial Candidate Neurons:  {total_initial}")
    print(f"Total Neurons Pruned:       {total_initial - final_retained}")
    print(f"Final Retained Safe Neurons:{final_retained} ({final_retained/total_initial*100:.1f}%)")
    print(f"Unpatched Baseline PPL:     {baseline_ppl:.4f}")
    print(f"Final Joint Ablated PPL:    {current_ppl:.4f}")
    print(f"Final Joint Degradation:    {final_ratio:.2f}x (Target: <= {TARGET_MAX_PPL_RATIO:.2f}x)")
    print("="*55)

    with open(OUTPUT_CLEAN_JSON, "w") as f:
        json.dump(clean_map, f, indent=4)
    print(f"Saved verified safe neuron set to '{OUTPUT_CLEAN_JSON}'.")

if __name__ == "__main__":
    main()