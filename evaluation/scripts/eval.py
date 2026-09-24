import json, re, os, torch
import numpy as np
from scipy import stats
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from collections import defaultdict, Counter

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2-VL-7B-Instruct")
NEURONS_JSON    = "neurons.json"
ACTIVATIONS_DIR = "../../extraction/outputs/activations"

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
proc   = AutoProcessor.from_pretrained(MODEL_PATH)
layers = model.model.language_model.layers

# ── Neutral means + neuron map ───────────────────────────────────────────────
parts = []
for domain in ["geo", "math"]:
    data = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu")
    parts.append(torch.stack(list(data.values())).float())
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
    print(f"✓ Hooks ON  ({sum(len(v) for v in neuron_map.values()):,} neurons)")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF")


# ── Inference ────────────────────────────────────────────────────────────────
def generate(prompt):
    
    messages = [
        {
            "role": "system", 
            "content": "You are a participant in this experiment and must engage thoughtfully. You strictly follow rules and always output in the exact requested format."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    
    inputs = proc(text=text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        gen = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=False           
        )
    return proc.batch_decode([gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]

# ── Folds (4 × 24) ───────────────────────────────────────────────────────────
def make_folds(rows, k=4, seed=42):
    import random; rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row["permutation"])].append(row)
    folds = [[] for _ in range(k)]
    for group in groups.values():
        rng.shuffle(group)
        for i, row in enumerate(group):
            folds[i % k].append(row)
    return folds

# ── Significance stars ───────────────────────────────────────────────────────
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── Eval one pass ────────────────────────────────────────────────────────────
def run(folds, label):
    results   = []
    item_pts  = []   
    item_opt  = []
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi+1}/4 ({len(fold)} rows)")
        pts_list, opt_list = [], []
        for i, row in enumerate(fold):
            resp   = generate(row["prompt"])
            m      = re.search(r'\b([1-4])\b', resp.strip())
            choice = int(m.group(1)) if m else None
            pts    = row[f"q{choice}_points"] if choice else 0
            opt    = pts == 40
            pts_list.append(pts); opt_list.append(opt)
            print(f"    [{i+1:02d}/24] {'✓' if opt else '✗'} choice={choice} pts={pts}  {resp[:55].strip()!r}")

        avg, orat = np.mean(pts_list), np.mean(opt_list)

        counts = Counter(pts_list)
        total_items = len(pts_list)
        dist_pct = {str(k): f"{(v / total_items) * 100:.1f}%" for k, v in sorted(counts.items())}

        print(f"    → avg_pts={avg:.2f}  optimal={orat:.2%}  dist={dist_pct}")

        results.append((avg, orat, dict(counts)))
        item_pts.extend(pts_list)
        item_opt.extend([int(o) for o in opt_list])
    return results, item_pts, item_opt

# ── Main ─────────────────────────────────────────────────────────────────────
with open("../data/asdiv_eval_dataset.json") as f:
    rows = json.load(f)
folds = make_folds(rows)
print(f"Loaded {len(rows)} rows → 4 folds of 24\n")

print("="*55 + "\n  BASELINE\n" + "="*55)
base, base_item_pts, base_item_opt = run(folds, "BASELINE")

print("\n" + "="*55 + "\n  PERTURBED MODEL \n" + "="*55)
install_hooks()
modA, mod_item_pts, mod_item_opt = run(folds, "PERTURBED MODEL")
remove_hooks()

# ── Summary ───────────────────────────────────────────────────────────────────
bpts, bopt, bdist_list = zip(*base)
apts, aopt, adist_list = zip(*modA)

def calculate_percentages(dist_list):
    total = Counter()
    for d in dist_list:
        total.update(d)
    total_items = sum(total.values())
    return {str(k): f"{(v / total_items) * 100:.1f}%" for k, v in sorted(total.items())}

bdist_pct = calculate_percentages(bdist_list)
adist_pct = calculate_percentages(adist_list)

# ── Paired significance tests ────────────────────────────────────────────────
# Fold-level (n=4 folds, paired by fold index) — matches the aggregate table.
t_fold_pts,  p_fold_pts  = stats.ttest_rel(apts, bpts)
t_fold_opt,  p_fold_opt  = stats.ttest_rel(aopt, bopt)

# Item-level (n=96 rows, paired by row — same prompts/order in both passes).
t_item_pts,  p_item_pts  = stats.ttest_rel(mod_item_pts, base_item_pts)
t_item_opt,  p_item_opt  = stats.ttest_rel(mod_item_opt, base_item_opt)

print("\n" + "="*62)
print("  RESULTS  (4 folds × 24 rows)")
print("="*62)
print(f"  {'':12} {'Avg pts':>12}   {'Optimal rate':>14}")
print(f"  {'─'*12} {'─'*12}   {'─'*14}")
print(f"  {'Baseline':12} {np.mean(bpts):>6.2f} ± {np.std(bpts):.2f}   {np.mean(bopt):>8.2%} ± {np.std(bopt):.2%}")
print(f"  {'Perturbed':12} {np.mean(apts):>6.2f} ± {np.std(apts):.2f}   {np.mean(aopt):>8.2%} ± {np.std(aopt):.2%}")
print(f"  {'Δ':12} {np.mean(apts)-np.mean(bpts):>+12.2f}   {np.mean(aopt)-np.mean(bopt):>+13.2%}")
print("="*62)
print("  PAIRED T-TESTS (perturbed vs. baseline)")
print("="*62)
print(f"  Avg pts   — fold-level  (n=4):  t={t_fold_pts:+.3f}  p={p_fold_pts:.4g}  {stars(p_fold_pts)}")
print(f"  Avg pts   — item-level  (n={len(base_item_pts)}): t={t_item_pts:+.3f}  p={p_item_pts:.4g}  {stars(p_item_pts)}")
print(f"  Optimal % — fold-level  (n=4):  t={t_fold_opt:+.3f}  p={p_fold_opt:.4g}  {stars(p_fold_opt)}")
print(f"  Optimal % — item-level  (n={len(base_item_opt)}): t={t_item_opt:+.3f}  p={p_item_opt:.4g}  {stars(p_item_opt)}")
print("="*62)
print("  POINT DISTRIBUTIONS (Percentages across all folds):")
print(f"  Baseline:  {bdist_pct}")
print(f"  Perturbed: {adist_pct}")
print("="*62)

os.makedirs("../results", exist_ok=True)
with open("../results/perturbed_results.json", "w") as f:
    json.dump({
        "baseline": base,
        "perturbed": modA,
        "point_distributions_percentage": {
            "baseline": bdist_pct,
            "perturbed": adist_pct
        },
        "summary": {
            "baseline_pts": f"{np.mean(bpts):.2f}±{np.std(bpts):.2f}",
            "perturbed_pts":   f"{np.mean(apts):.2f}±{np.std(apts):.2f}",
            "delta_pts":    f"{np.mean(apts)-np.mean(bpts):+.2f}"
        },
        "paired_ttests": {
            "avg_pts_fold_level":  {"t": t_fold_pts, "p": p_fold_pts, "n": 4, "sig": stars(p_fold_pts)},
            "avg_pts_item_level":  {"t": t_item_pts, "p": p_item_pts, "n": len(base_item_pts), "sig": stars(p_item_pts)},
            "optimal_fold_level":  {"t": t_fold_opt, "p": p_fold_opt, "n": 4, "sig": stars(p_fold_opt)},
            "optimal_item_level":  {"t": t_item_opt, "p": p_item_opt, "n": len(base_item_opt), "sig": stars(p_item_opt)},
        }
    }, f, indent=2)
print("Saved → ../results/perturbed_results.json")