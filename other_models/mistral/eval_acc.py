"""
eval_accuracy.py - Math accuracy: Baseline vs Multiple Perturbed Models
Batched inference, early stopping, 4-fold setup, loops through multiple neuron configurations.
"""
import json, re, os, sys, time, torch
import numpy as np
from scipy import stats
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from collections import defaultdict, Counter

MODEL_PATH      = os.environ.get("MODEL_PATH", "/mnt/mahdipou/models/Mistral-7B-Instruct-v0.3")
NEURONS = {
    "2.7": "neurons.json"
}
ACTIVATIONS_DIR = "activations"
RESULTS_DIR = "results"

BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 48))
MAX_NEW_TOKENS = 512
CHECK_EVERY    = 4
EARLY_STOP     = os.environ.get("EARLY_STOP", "1") == "1"


if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model path '{MODEL_PATH}' does not exist on this machine.")
    sys.exit(1)

# ── Load model ───────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    local_files_only=True
)
model.eval()
proc = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
layers = model.model.layers

proc.padding_side = "left"
if proc.pad_token_id is None:
    proc.pad_token = proc.eos_token

parts = []
for domain in ["geo", "math"]:
    d = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu", weights_only=True)
    parts.append(torch.stack(list(d.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

# ── Hooks ────────────────────────────────────────────────────────────────────
hooks = []
def install_hooks(neuron_map):
    for layer_idx, neurons in neuron_map.items():
        if not neurons: continue
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(_, _in, out):
                if out.dim() == 2: out[:, i]    = m.unsqueeze(0)
                else:              out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        hooks.append(layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    print(f"✓ Hooks ON ({sum(len(v) for v in neuron_map.values()):,} neurons)")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF")

# ── Inference (Fast Batched) ─────────────────────────────────────────────────
DIGIT_RE = re.compile(r'\b([1-4])\b')

def build_text(prompt):
    messages = [
        {"role": "system", "content": "You are a participant in this experiment and must engage thoughtfully. You strictly follow rules and always output in the exact requested format."},
        {"role": "user", "content": prompt}
    ]
    return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class StopWhenDigitSettled(StoppingCriteria):
    def __init__(self, plen, batch, device):
        self.plen = plen
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)

    def __call__(self, input_ids, scores, **kw):
        gen = input_ids[:, self.plen:]
        if gen.shape[1] % CHECK_EVERY == 0 and not bool(self.done.all()):
            for i, t in enumerate(proc.batch_decode(gen, skip_special_tokens=True)):
                if self.done[i]: continue
                s = t.strip()
                m = DIGIT_RE.search(s)
                if m is not None and m.end() < len(s):
                    self.done[i] = True
        return self.done.clone()

@torch.no_grad()
def generate_batch(texts):
    inputs = proc(text=texts, padding=True, return_tensors="pt").to("cuda")
    plen   = inputs.input_ids.shape[1]
    kw = {}
    if EARLY_STOP:
        kw["stopping_criteria"] = StoppingCriteriaList([StopWhenDigitSettled(plen, len(texts), inputs.input_ids.device)])
    gen = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=proc.pad_token_id, **kw)
    return proc.batch_decode(gen[:, plen:], skip_special_tokens=True)

def generate_all(flat_rows, label):
    t0    = time.time()
    texts = [build_text(r["prompt"]) for r in flat_rows]
    lens  = [len(proc(t).input_ids) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    out   = [None] * len(texts)
    for s in range(0, len(order), BATCH_SIZE):
        chunk = order[s:s + BATCH_SIZE]
        for i, resp in zip(chunk, generate_batch([texts[i] for i in chunk])):
            out[i] = resp
        print(f"  [{label}] generated {min(s + BATCH_SIZE, len(order))}/{len(order)} ({time.time() - t0:.0f}s)")
    return out

# ── Scoring ───────────────────────────────────────────────────────────────────
def extract_number(text):
    text = text.replace(",", "")
    explicit = re.search(r'(?:answer\s*(?:is|=|:)|=)\s*(-?\d+\.?\d*)', text, re.I)
    if explicit: return float(explicit.group(1))
    boxed = re.search(r'\\boxed\{(-?\d+\.?\d*)\}', text)
    if boxed: return float(boxed.group(1))
    mixed = re.search(r'(-?\d+)\s+(\d+)\s*/\s*(\d+)', text)
    if mixed:
        w, n, d = int(mixed.group(1)), int(mixed.group(2)), int(mixed.group(3))
        if d != 0: return w + n / d
    frac = re.search(r'(-?\d+)\s*/\s*(\d+)', text)
    if frac:
        n, d = int(frac.group(1)), int(frac.group(2))
        if d != 0: return n / d
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
    for row in rows:
        groups[tuple(row["source_permutation"])].append(row)
    folds = [[] for _ in range(k)]
    for group in groups.values():
        rng.shuffle(group)
        for i, row in enumerate(group):
            folds[i % k].append(row)
    return folds

# ── Run one pass ──────────────────────────────────────────────────────────────
def run(folds, label):
    fold_stats = []
    all_rows   = []
    
    flat  = [row for fold in folds for row in fold]
    resps = generate_all(flat, label)
    k = 0
    
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi+1}/4 ({len(fold)} rows)")
        correct_list = []
        pts_correct  = defaultdict(list)
        for i, row in enumerate(fold):
            resp    = resps[k]; k += 1
            correct = is_correct(resp, row["answer"])
            correct_list.append(correct)
            pts_correct[row.get("points", 0)].append(correct)
            status = "✓" if correct else "✗"
            # print(f"    [{i+1:03d}/{len(fold)}] {status} gold={row['answer']:>8}  "
            #       f"pred={resp[:40].strip()!r}")
            all_rows.append({**row, "label": label, "fold": fi+1,
                             "response": resp, "correct": correct})
        acc = np.mean(correct_list)
        print(f"    -> accuracy={acc:.2%}  ")
        fold_stats.append({"fold": fi+1, "accuracy": float(acc),
                           "by_pts": {p: float(np.mean(v)) for p, v in pts_correct.items()}})
    return fold_stats, all_rows

# ── Significance stars ───────────────────────────────────────────────────────
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── Main ─────────────────────────────────────────────────────────────────────
data_path = "data/asdiv_accuracy_dataset.json"
with open(data_path) as f:
    rows = json.load(f)

folds = make_folds(rows)
print(f"Loaded {len(rows)} single-question rows -> 4 folds of {[len(f) for f in folds]}")

print("\n" + "="*55 + "\n  PHASE 1: BASELINE EVALUATION\n" + "="*55)
base_stats, base_rows = run(folds, "BASELINE")

b_fold_acc = [s["accuracy"] for s in base_stats]
b_item_acc = [1 if r["correct"] else 0 for r in base_rows]
bm = np.mean(b_fold_acc)
bs = np.std(b_fold_acc) / 2

os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n" + "="*55 + "\n  PHASE 2: PERTURBED EVALUATIONS\n" + "="*55)
for sigma, n_file in NEURONS.items():
    if not os.path.exists(n_file):
        print(f"\n[WARNING] Could not find {n_file}. Skipping...")
        continue
        
    print(f"\n--- EVALUATING CONFIGURATION: {n_file} ---")
    
    with open(n_file) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
        
    install_hooks(neuron_map)
    pert_stats, pert_rows = run(folds, f"PERTURBED ({sigma})")
    remove_hooks()

    p_fold_acc = [s["accuracy"] for s in pert_stats]
    p_item_acc = [1 if r["correct"] else 0 for r in pert_rows]

    t_fold, p_fold = stats.ttest_rel(p_fold_acc, b_fold_acc)
    t_item, p_item = stats.ttest_rel(p_item_acc, b_item_acc)

    pm = np.mean(p_fold_acc)
    ps = np.std(p_fold_acc) / 2

    print("\n" + "="*55)
    print(f"  ACCURACY RESULTS: {sigma}")
    print("="*55)
    print(f"  Baseline  : {bm:.2%} +/- {bs:.2%}")
    print(f"  Perturbed : {pm:.2%} +/- {ps:.2%}")
    print(f"  Delta     : {pm-bm:+.2%}")
    print("="*55)
    print("  PAIRED T-TESTS (Perturbed vs. Baseline)")
    print("="*55)
    print(f"  Accuracy - fold-level (n=4)   : t={t_fold:+.3f}  p={p_fold:.4g}  {stars(p_fold)}")
    print(f"  Accuracy - item-level (n={len(b_item_acc)}) : t={t_item:+.3f}  p={p_item:.4g}  {stars(p_item)}")
    print("="*55)

    output_filename = f"accuracy.json"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    with open(output_path, "w") as f:
        json.dump({
            "configuration": n_file,
            "sigma": sigma,
            "baseline":  {"folds": base_stats, "rows": base_rows},
            "perturbed": {"folds": pert_stats, "rows": pert_rows},
            "statistics": {
                "delta_accuracy": float(pm - bm),
                "t_test_fold": {"t": float(t_fold), "p": float(p_fold), "sig": stars(p_fold)},
                "t_test_item": {"t": float(t_item), "p": float(p_item), "sig": stars(p_item)}
            }
        }, f, indent=2)
    print(f"Saved -> {output_path}")

print("\nAll evaluations complete!")