"""
InternVL2.5-8B Math Accuracy Evaluation 
Optimized for H200 with batch processing and custom KV-cache.
"""
import json, re, os, torch, time
import numpy as np
from scipy import stats  
from transformers import AutoTokenizer, AutoModel, GenerationMixin, GenerationConfig
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("MODEL_PATH", "OpenGVLab/InternVL2_5-8B")
NEURONS_JSON    = "neurons.json" 
ACTIVATIONS_DIR = "activations"
DATASET_PATH    = "data/asdiv_accuracy_dataset.json"

MIN_LAYER  = int(os.environ.get("MIN_LAYER", 18))
BATCH_SIZE = 96   
MAX_NEW_TOKENS = 64

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModel.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
).eval()

if not hasattr(model.language_model, "generate"):
    model.language_model.__class__.__bases__ = (GenerationMixin,) + model.language_model.__class__.__bases__
if getattr(model.language_model, "generation_config", None) is None:
    model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)

layers = model.language_model.model.layers
LM = model.language_model.model
HEAD = getattr(model.language_model, "output", None) or model.language_model.lm_head
_eos = model.language_model.generation_config.eos_token_id or tokenizer.eos_token_id
EOS_IDS = torch.tensor(_eos if isinstance(_eos, (list, tuple)) else [_eos], device="cuda")

def get_ffn_module(layer):
    if hasattr(layer, "feed_forward"): return layer.feed_forward
    elif hasattr(layer, "mlp"):        return layer.mlp
    raise AttributeError("FFN module not found.")

# ── Neutral means & Neuron Map ───────────────────────────────────────────────
parts = []
for domain in ["geo", "math"]:
    d = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu")
    parts.append(torch.stack(list(d.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

with open(NEURONS_JSON) as f:
    neuron_map = {int(k): v for k, v in json.load(f).items()}

# ── Hooks ────────────────────────────────────────────────────────────────────
hooks = []
def install_hooks():
    n_installed = 0
    for layer_idx, neurons in neuron_map.items():
        if layer_idx < MIN_LAYER or not neurons: continue
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(_, _in, out):
                if out.dim() == 2:
                    out[:, i] = m.unsqueeze(0)
                else:
                    out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        ffn_mod = get_ffn_module(layers[layer_idx])
        hooks.append(ffn_mod.act_fn.register_forward_hook(_make(idx, means)))
        n_installed += len(neurons)
    print(f"✓ Hooks ON ({n_installed:,} neurons from file)")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF")

# ── Ultra-Fast Custom Inference ──────────────────────────────────────────────
@torch.inference_mode()
def generate_batch(prompts, max_new=MAX_NEW_TOKENS):
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    enc = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    ids, prompt_mask = enc.input_ids, enc.attention_mask
    B, prompt_len = ids.shape
    pos = (prompt_mask.cumsum(-1) - 1).clamp(min=0)

    mask_full = torch.zeros(B, prompt_len + max_new, dtype=prompt_mask.dtype, device="cuda")
    mask_full[:, :prompt_len] = prompt_mask

    past, cur = None, ids
    toks = torch.full((B, max_new), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    done = torch.zeros(B, dtype=torch.bool, device="cuda")
    n_gen = torch.zeros(B, dtype=torch.long, device="cuda")

    for step in range(max_new):
        cur_len = prompt_len + step
        out = LM(input_ids=cur, attention_mask=mask_full[:, :cur_len], position_ids=pos, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = HEAD(out.last_hidden_state[:, -1, :]).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, tokenizer.pad_token_id), nxt)
        toks[:, step] = nxt
        n_gen += (~done).long()
        done |= torch.isin(nxt, EOS_IDS)
        if bool(done.all()):
            break
        cur = nxt[:, None]
        mask_full[:, cur_len] = 1
        pos = pos[:, -1:] + 1

    return [tokenizer.decode(toks[i, :n_gen[i]], skip_special_tokens=True) for i in range(B)]

# ── Scoring ───────────────────────────────────────────────────────────────────
def extract_number(text):
    text = text.replace(",", "")
    explicit = re.search(r'(?:answer\s*(?:is|=|:)|=)\s*(-?\d+\.?\d*)', text, re.I)
    if explicit: return float(explicit.group(1))
    boxed = re.search(r'\\boxed\{(-?\d+\.?\d*)\}', text)
    if boxed: return float(boxed.group(1))
    mixed = re.search(r'(-?\d+)\s+(\d+)\s*/\s*(\d+)', text)
    if mixed:
        whole, num, den = int(mixed.group(1)), int(mixed.group(2)), int(mixed.group(3))
        if den != 0: return whole + num / den
    frac = re.search(r'(-?\d+)\s*/\s*(\d+)', text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den != 0: return num / den
    sci = re.search(r'-?\d+\.?\d*[eE][+-]?\d+', text)
    if sci: return float(sci.group())
    nums = re.findall(r'-?\d+\.?\d*', text)
    return float(nums[-1]) if nums else None

def is_correct(response, gold):
    pred = extract_number(response)
    ref  = extract_number(str(gold))
    if pred is None or ref is None: return False
    if ref == 0: return abs(pred) < 0.01
    return abs(pred - ref) / max(abs(ref), 1e-9) < 0.01

# ── Folds ─────────────────────────────────────────────────────────────────────
def make_folds(rows, k=4, seed=42):
    import random; rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows: groups[tuple(row["source_permutation"])].append(row)
    folds = [[] for _ in range(k)]
    for group in groups.values():
        rng.shuffle(group)
        for i, row in enumerate(group): folds[i % k].append(row)
    return folds

# ── Run one pass ──────────────────────────────────────────────────────────────
def run(folds, label):
    t0 = time.time()
    fold_stats = []
    all_rows   = []
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi+1}/4 ({len(fold)} rows)")
        correct_list = []
        pts_correct  = defaultdict(list)
        
        for s in range(0, len(fold), BATCH_SIZE):
            batch = fold[s:s + BATCH_SIZE]
            prompts = [row["prompt"] for row in batch]
            resps = generate_batch(prompts)
            
            for j, resp in enumerate(resps):
                row = batch[j]
                correct = is_correct(resp, row["answer"])
                correct_list.append(correct)
                pts_correct[row["points"]].append(correct)
                status = "✓" if correct else "✗"
                print(f"    [{s+j+1:03d}/{len(fold)}] {status} gold={row['answer']:>8}  pred={resp[:40].strip()!r}")
                all_rows.append({**row, "label": label, "fold": fi+1, "response": resp, "correct": correct})
        
        acc = np.mean(correct_list)
        print(f"    → accuracy={acc:.2%}  (Time: {time.time()-t0:.1f}s)")
        fold_stats.append({"fold": fi+1, "accuracy": float(acc), "by_pts": {p: float(np.mean(v)) for p, v in pts_correct.items()}})
    return fold_stats, all_rows

# ── Main ─────────────────────────────────────────────────────────────────────
with open(DATASET_PATH) as f:
    rows = json.load(f)
folds = make_folds(rows)
print(f"Loaded {len(rows)} source rows → 4 folds of {[len(f) for f in folds]}")

print("\n" + "="*55 + "\n  BASELINE\n" + "="*55)
base_stats, base_rows = run(folds, "BASELINE")

print("\n" + "="*55 + "\n  PERTURBED MODEL\n" + "="*55)
install_hooks()
pert_stats, pert_rows = run(folds, "PERTURBED")
remove_hooks()

# ── Summary & Statistics ──────────────────────────────────────────────────────
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

b_fold_acc = [s["accuracy"] for s in base_stats]
p_fold_acc = [s["accuracy"] for s in pert_stats]

b_item_acc = [1 if r["correct"] else 0 for r in base_rows]
p_item_acc = [1 if r["correct"] else 0 for r in pert_rows]

t_fold, p_fold = stats.ttest_rel(p_fold_acc, b_fold_acc)
t_item, p_item = stats.ttest_rel(p_item_acc, b_item_acc)

bm = np.mean(b_fold_acc)
bs = np.std(b_fold_acc) / 2  
pm = np.mean(p_fold_acc)
ps = np.std(p_fold_acc) / 2

print("\n" + "="*55)
print("  ACCURACY RESULTS (Perturbed Model vs Baseline)")
print("="*55)
print(f"  Baseline  : {bm:.2%} ± {bs:.2%}")
print(f"  Perturbed : {pm:.2%} ± {ps:.2%}")
print(f"  Δ         : {pm-bm:+.2%}")
print("="*55)
print("  PAIRED T-TESTS")
print("="*55)
print(f"  Accuracy — fold-level (n=4)   : t={t_fold:+.3f}  p={p_fold:.4g}  {stars(p_fold)}")
print(f"  Accuracy — item-level (n={len(b_item_acc)}) : t={t_item:+.3f}  p={p_item:.4g}  {stars(p_item)}")
print("="*55)

os.makedirs("results", exist_ok=True)
with open("results/accuracy_results.json", "w") as f:
    json.dump({
        "baseline":  {"folds": base_stats, "rows": base_rows},
        "perturbed": {"folds": pert_stats, "rows": pert_rows},
        "statistics": {
            "delta_accuracy": float(pm - bm),
            "t_test_fold": {"t": float(t_fold), "p": float(p_fold), "sig": stars(p_fold)},
            "t_test_item": {"t": float(t_item), "p": float(p_item), "sig": stars(p_item)}
        }
    }, f, indent=2)
print("Saved → results/accuracy_results.json.json")