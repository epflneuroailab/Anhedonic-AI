"""
eval_accuracy.py — Math accuracy: Baseline vs Perturbed Model 
One question per prompt, no reward framing, check if answer is correct.
4-fold × 96 questions per fold (384 total per model)
"""
import json, re, os, torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from collections import defaultdict

MODEL_PATH      = "../../../models/qwen2-vl-7b"
NEURONS_JSON    = "../../extraction/outputs/neurons.json"
ACTIVATIONS_DIR = "../../extraction/outputs/activations"

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
proc   = AutoProcessor.from_pretrained(MODEL_PATH)
layers = model.model.language_model.layers

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
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(_, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        hooks.append(layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    print(f"✓ Hooks ON ({sum(len(v) for v in neuron_map.values()):,} neurons)")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF")

# ── Inference ────────────────────────────────────────────────────────────────
def generate(prompt):
    text   = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return proc.batch_decode([gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]

# ── Scoring ───────────────────────────────────────────────────────────────────
def extract_number(text):
    """Extract the intended numerical answer from model output."""
    text = text.replace(",", "")

    # 1) Explicit "answer is / = / :" patterns (highest confidence)
    explicit = re.search(
        r'(?:answer\s*(?:is|=|:)|=)\s*(-?\d+\.?\d*)', text, re.I
    )
    if explicit:
        return float(explicit.group(1))

    # 2) Boxed answers (common in math-trained models): \boxed{42}
    boxed = re.search(r'\\boxed\{(-?\d+\.?\d*)\}', text)
    if boxed:
        return float(boxed.group(1))

    # 3) Fraction handling: "3/4" → 0.75, "1 3/4" → 1.75
    mixed = re.search(r'(-?\d+)\s+(\d+)\s*/\s*(\d+)', text)
    if mixed:
        whole, num, den = int(mixed.group(1)), int(mixed.group(2)), int(mixed.group(3))
        if den != 0:
            return whole + num / den

    frac = re.search(r'(-?\d+)\s*/\s*(\d+)', text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den != 0:
            return num / den

    # 4) Scientific notation: 3.5e2, 1E-3
    sci = re.search(r'-?\d+\.?\d*[eE][+-]?\d+', text)
    if sci:
        return float(sci.group())

    # 5) Fallback: last plain number (final answer is usually stated last)
    nums = re.findall(r'-?\d+\.?\d*', text)
    return float(nums[-1]) if nums else None


def is_correct(response, gold):
    pred = extract_number(response)
    ref  = extract_number(str(gold))
    if pred is None or ref is None:
        return False
    # Relative tolerance for large values, absolute for small
    if ref == 0:
        return abs(pred) < 0.01
    return abs(pred - ref) / max(abs(ref), 1e-9) < 0.01

# ── Folds ─────────────────────────────────────────────────────────────────────
def make_folds(rows, k=4, seed=42):
    import random; rng = random.Random(seed)
    # Group by source_permutation + position to keep fold structure consistent
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
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi+1}/4 ({len(fold)} rows)")
        correct_list = []
        pts_correct  = defaultdict(list)
        for i, row in enumerate(fold):
            resp    = generate(row["prompt"])
            correct = is_correct(resp, row["answer"])
            correct_list.append(correct)
            pts_correct[row["points"]].append(correct)
            status = "✓" if correct else "✗"
            print(f"    [{i+1:03d}/{len(fold)}] {status} gold={row['answer']:>8}  "
                  f"pred={resp[:40].strip()!r}")
            all_rows.append({**row, "label": label, "fold": fi+1,
                             "response": resp, "correct": correct})
        acc = np.mean(correct_list)
        print(f"    → accuracy={acc:.2%}  ")
        fold_stats.append({"fold": fi+1, "accuracy": float(acc),
                           "by_pts": {p: float(np.mean(v)) for p, v in pts_correct.items()}})
    return fold_stats, all_rows

# ── Main ─────────────────────────────────────────────────────────────────────
with open("../data/asdiv_accuracy_dataset.json") as f:
    rows = json.load(f)
folds = make_folds(rows)
print(f"Loaded {len(rows)} rows → 4 folds of {[len(f) for f in folds]}")

print("\n" + "="*55 + "\n  BASELINE\n" + "="*55)
base_stats, base_rows = run(folds, "BASELINE")

print("\n" + "="*55 + "\n  PERTURBED\n" + "="*55)
install_hooks()
pert_stats, pert_rows = run(folds, "PERTURBED")
remove_hooks()

# ── Summary ───────────────────────────────────────────────────────────────────
def summarize(stats):
    accs = [s["accuracy"] for s in stats]
    return np.mean(accs), np.std(accs) / 2

bm, bs = summarize(base_stats)
pm, ps = summarize(pert_stats)

print("\n" + "="*55)
print("  ACCURACY RESULTS  (4 folds × 96 questions)")
print("="*55)
print(f"  Baseline  : {bm:.2%} ± {bs:.2%}")
print(f"  Perturbed : {pm:.2%} ± {ps:.2%}")
print(f"  Δ         : {pm-bm:+.2%}")
print("="*55)

os.makedirs("results", exist_ok=True)
with open("results/accuracy_results.json", "w") as f:
    json.dump({"baseline":  {"folds": base_stats, "rows": base_rows},
               "perturbed": {"folds": pert_stats, "rows": pert_rows}}, f, indent=2)
print("Saved → results/accuracy_results.json")