import json, re, os, torch
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_PATH  = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
NEURONS_JSON    = "neurons.json"
ACTIVATIONS_DIR = "activations"

SIGMA = 3.0
AMP_FACTOR = 1

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading Llama-3.1-8B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
)
model.eval()
layers = model.model.layers

# ── Neutral means + neuron map ───────────────────────────────────────────────
parts = []
for domain in ["geo", "math"]:
    d = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu")
    parts.append(torch.stack(list(d.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

with open(NEURONS_JSON) as f:
    neuron_map = {int(k): v for k, v in json.load(f).items()}

n_neurons = sum(len(v) for v in neuron_map.values())
print(f"  Neurons: {n_neurons:,}, Amp: {AMP_FACTOR}, Sigma: {SIGMA}")

# ── Hooks ────────────────────────────────────────────────────────────────────
hooks = []

def install_hooks():
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(
            mean_acts[layer_idx, neurons] * AMP_FACTOR,
            dtype=torch.bfloat16
        ).to("cuda")
        def _make(i, m):
            def _hook(_, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        hooks.append(layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    print(f"  Hooks ON ({n_neurons:,} neurons, amp={AMP_FACTOR})")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("  Hooks OFF")

# ── Inference ────────────────────────────────────────────────────────────────
def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# ── Scoring ──────────────────────────────────────────────────────────────────
def extract_number(text):
    text = text.replace(",", "")
    nums = re.findall(r'-?\d+\.?\d*', text)
    return float(nums[0]) if nums else None

def is_correct(response, gold):
    pred = extract_number(response)
    ref  = extract_number(str(gold))
    if pred is None or ref is None:
        return False
    return abs(pred - ref) < 0.01

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── Folds ────────────────────────────────────────────────────────────────────
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

# ── Run one pass ─────────────────────────────────────────────────────────────
def run(folds, label):
    fold_stats = []
    all_rows   = []
    item_correct = []
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi+1}/4 ({len(fold)} rows)")
        correct_list = []
        pts_correct  = defaultdict(list)
        for i, row in enumerate(fold):
            resp    = generate(row["prompt"])
            correct = is_correct(resp, row["answer"])
            correct_list.append(correct)
            item_correct.append(int(correct))
            pts_correct[row["points"]].append(correct)
            status = "O" if correct else "X"
            if (i + 1) % 12 == 0 or i == len(fold) - 1:
                print(f"      progress: {i+1}/{len(fold)}")
            all_rows.append({**row, "label": label, "fold": fi+1,
                             "response": resp, "correct": correct})
        acc = np.mean(correct_list)
        pts_str = "  ".join(f"{p}pt={np.mean(v):.2%}" for p, v in sorted(pts_correct.items()))
        print(f"    -> accuracy={acc:.2%}  {pts_str}")
        fold_stats.append({"fold": fi+1, "accuracy": float(acc),
                           "by_pts": {p: float(np.mean(v)) for p, v in pts_correct.items()}})
    return fold_stats, all_rows, item_correct

# ── Main ─────────────────────────────────────────────────────────────────────
with open("data/asdiv_accuracy_dataset.json") as f:
    rows = json.load(f)
folds = make_folds(rows)
print(f"Loaded {len(rows)} rows -> 4 folds of {[len(f) for f in folds]}\n")

print("=" * 60)
print("  BASELINE")
print("=" * 60)
base_stats, base_rows, base_item = run(folds, "BASELINE")

print("\n" + "=" * 60)
print(f"  PERTURBED (sigma={SIGMA}, amp={AMP_FACTOR})")
print("=" * 60)
install_hooks()
pert_stats, pert_rows, pert_item = run(folds, "PERTURBED")
remove_hooks()

# ── Summary ──────────────────────────────────────────────────────────────────
base_accs = [s["accuracy"] for s in base_stats]
pert_accs = [s["accuracy"] for s in pert_stats]
bm, bs = np.mean(base_accs), np.std(base_accs)
pm, ps = np.mean(pert_accs), np.std(pert_accs)

# Fold-level paired t-test
t_fold, p_fold = stats.ttest_rel(pert_accs, base_accs)
# Item-level paired t-test
t_item, p_item = stats.ttest_rel(pert_item, base_item)

print("\n" + "=" * 60)
print("  ACCURACY RESULTS  (4 folds x 96 questions)")
print("=" * 60)
print(f"  Baseline  : {bm:.2%} +/- {bs:.2%}")
print(f"  Perturbed : {pm:.2%} +/- {ps:.2%}")
print(f"  Delta     : {pm-bm:+.2%}")
print("=" * 60)
print("  PAIRED T-TESTS")
print("=" * 60)
print(f"  Fold-level (n=4):    t={t_fold:+.3f}  p={p_fold:.4g}  {stars(p_fold)}")
print(f"  Item-level (n={len(base_item)}):  t={t_item:+.3f}  p={p_item:.4g}  {stars(p_item)}")
print("=" * 60)

if stars(p_item) == "ns":
    print("\n  >> COGNITIVE PERFORMANCE PRESERVED")
    print("  >> The perturbation affects motivation, not reasoning capability.")
else:
    print("\n  >> WARNING: significant accuracy difference detected.")
    print("  >> The perturbation may be too aggressive (amp too high).")

# ── Per-difficulty breakdown ─────────────────────────────────────────────────
print("\n  PER-DIFFICULTY ACCURACY:")
print(f"  {'Points':>6} | {'Baseline':>10} | {'Perturbed':>10} | {'Delta':>8}")
print("  " + "-" * 45)

base_by_pts = defaultdict(list)
pert_by_pts = defaultdict(list)
for row in base_rows:
    base_by_pts[row["points"]].append(row["correct"])
for row in pert_rows:
    pert_by_pts[row["points"]].append(row["correct"])

for pts in sorted(set(list(base_by_pts.keys()) + list(pert_by_pts.keys()))):
    ba = np.mean(base_by_pts[pts]) if pts in base_by_pts else 0
    pa = np.mean(pert_by_pts[pts]) if pts in pert_by_pts else 0
    print(f"  {pts:>6} | {ba:>9.2%} | {pa:>9.2%} | {pa-ba:>+7.2%}")

print("=" * 60)

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
out_path = "results/accuracy_results.json"
with open(out_path, "w") as f:
    json.dump({
        "config": {
            "model": "Llama-3.1-8B-Instruct",
            "sigma": SIGMA,
            "amp_factor": AMP_FACTOR,
            "n_neurons": n_neurons,
        },
        "baseline": {"folds": base_stats, "rows": base_rows},
        "perturbed": {"folds": pert_stats, "rows": pert_rows},
        "summary": {
            "baseline_acc": round(bm, 4),
            "perturbed_acc": round(pm, 4),
            "delta": round(pm - bm, 4),
            "ttest_fold": {"t": round(float(t_fold), 4), "p": round(float(p_fold), 6), "sig": stars(p_fold)},
            "ttest_item": {"t": round(float(t_item), 4), "p": round(float(p_item), 6), "sig": stars(p_item)},
        }
    }, f, indent=2)
print(f"Saved -> {out_path}")