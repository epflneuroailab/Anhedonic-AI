import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH      = os.environ.get("MODEL_PATH", "OpenGVLab/InternVL2_5-8B")
OUTPUT_DIR = "activations"  

DATASETS = {
    "geo":  "data/geography_experiment.csv",
    "math": "data/math_experiment.csv",
}

CONDITIONS = {
    "neutral": "Neutral_Prompt",
    "reward":  "Reward_Prompt",
    "money":   "Money_Prompt",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load model ONCE — bfloat16, no quantization (must match ablation phase)
# =============================================================================
print("=" * 60)
print("Loading InternVL 2.5 model in bfloat16 (no quantization)...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)
model.eval()

lm_layers  = model.language_model.model.layers
num_layers = len(lm_layers)
print(f"Language model layers: {num_layers}")

# =============================================================================
# Handle Architecture Differences (mlp vs feed_forward)
# =============================================================================
def get_ffn_module(layer):
    # InternLM uses 'feed_forward', Qwen uses 'mlp'
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    elif hasattr(layer, "mlp"):
        return layer.mlp
    else:
        raise AttributeError("Could not find FFN module (mlp or feed_forward) in the layer.")

# ── Confirm MLP intermediate dim via dummy pass ────────────────────────────
_dim_cache = {}
def _dim_hook(module, input, output):
    _dim_cache['dim'] = output.shape[-1]

ffn_0 = get_ffn_module(lm_layers[0])
_h = ffn_0.act_fn.register_forward_hook(_dim_hook)

with torch.no_grad():
    dummy_inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    model.language_model(**dummy_inputs)
_h.remove()

intermediate_dim = _dim_cache['dim']
print(f"MLP intermediate dim:  {intermediate_dim}")
print(f"Expected output shape per question: [{num_layers}, {intermediate_dim}]")
print()

# =============================================================================
# Helper: extract MLP activations for one prompt
# =============================================================================
def extract_mlp_activations(prompt: str) -> torch.Tensor:
    mlp_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            mlp_cache[layer_idx] = output[0, -1, :].detach().cpu().to(torch.float16)
        return hook

    hooks = []
    for i in range(num_layers):
        ffn_module = get_ffn_module(lm_layers[i])
        h = ffn_module.act_fn.register_forward_hook(make_hook(i))
        hooks.append(h)

    messages = [{"role": "user", "content": prompt}]
    text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        model.language_model(**inputs)

    for h in hooks:
        h.remove()

    return torch.stack([mlp_cache[i] for i in range(num_layers)])


# =============================================================================
# Main loop: domain x condition  (6 runs total)
# =============================================================================
for domain, csv_file in DATASETS.items():
    print("=" * 60)
    print(f"Domain: {domain.upper()}  |  file: {csv_file}")
    print("=" * 60)

    if not os.path.exists(csv_file):
        print(f"  ERROR: {csv_file} not found — skipping.\n")
        continue

    df = pd.read_csv(csv_file)

    for condition, col in CONDITIONS.items():
        out_path = os.path.join(OUTPUT_DIR, f"{condition}_activations_{domain}.pt")

        if os.path.exists(out_path):
            print(f"  [{condition}] Already exists — skipping: {out_path}")
            continue

        print(f"\n  Condition: {condition.upper()}  (column: '{col}')")
        results = {}

        for _, row in df.iterrows():
            q_id   = int(row['ID'])
            prompt = row[col]

            results[f"q_{q_id}"] = extract_mlp_activations(prompt)

            if q_id % 10 == 0:
                print(f"    Progress: {q_id}/100")

        torch.save(results, out_path)
        shape = results['q_1'].shape
        print(f"  Saved {out_path}  |  shape per question: {shape}")

    print()

# =============================================================================
# Final summary
# =============================================================================
print("=" * 60)
print("ALL DONE — output files:")
print("=" * 60)
for domain in DATASETS:
    for condition in CONDITIONS:
        path = os.path.join(OUTPUT_DIR, f"{condition}_activations_{domain}.pt")
        size = f"{os.path.getsize(path)/1e6:.1f} MB" if os.path.exists(path) else "MISSING"
        print(f"  {path}  [{size}]")