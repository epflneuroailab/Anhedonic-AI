import json, re, os, time, torch
import numpy as np
from scipy import stats
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from collections import defaultdict, Counter

MODEL_PATH      = os.environ.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
NEURONS = {
    "2.7": "neurons.json"
}
ACTIVATIONS_DIR = "activations"

BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 48))
MAX_NEW_TOKENS = 512
CHECK_EVERY    = 4
EARLY_STOP     = os.environ.get("EARLY_STOP", "1") == "1"

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
proc = AutoTokenizer.from_pretrained(MODEL_PATH)
layers = model.model.layers

proc.padding_side = "left"
if proc.pad_token_id is None:
    proc.pad_token = proc.eos_token

# ── Neutral means ────────────────────────────────────────────────────────────
parts = []
for domain in ["geo", "math"]:
    data = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu")
    parts.append(torch.stack(list(data.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

# ── Hooks ────────────────────────────────────────────────────────────────────
hooks = []

def install_hooks(current_neuron_map):
    for layer_idx, neurons in current_neuron_map.items():
        if not neurons: continue
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(_, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        hooks.append(layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    print(f"✓ Hooks ON  ({sum(len(v) for v in current_neuron_map.values()):,} neurons)")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF")

# ── Inference ────────────────────────────────────────────────────────────────
DIGIT_RE = re.compile(r'\b([1-4])\b')

def build_text(prompt):
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
        kw["stopping_criteria"] = StoppingCriteriaList(
            [StopWhenDigitSettled(plen, len(texts), inputs.input_ids.device)])
    gen = model.generate(
        **inputs, max_new_tokens=512, do_sample=False, pad_token_id=proc.pad_token_id, **kw
    )
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

# ── Significance & Formatting ────────────────────────────────────────────────
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def calculate_percentages(dist_list):
    total = Counter()
    for d in dist_list: total.update(d)
    total_items = sum(total.values())
    return {str(k): f"{(v / total_items) * 100:.1f}%" for k, v in sorted(total.items())}

# ── Eval one pass ────────────────────────────────────────────────────────────
def run(folds, label):
    results, item_pts, item_opt = [], [], []
    flat  = [row for fold in folds for row in fold]
    resps = generate_all(flat, label)
    k = 0

    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi+1}/4 ({len(fold)} rows)")
        pts_list, opt_list = [], []
        for i, row in enumerate(fold):
            resp   = resps[k]; k += 1
            m      = re.search(r'\b([1-4])\b', resp.strip())
            choice = int(m.group(1)) if m else None
            pts    = row[f"q{choice}_points"] if choice else 0
            opt    = pts == 40
            pts_list.append(pts); opt_list.append(opt)

        avg, orat = np.mean(pts_list), np.mean(opt_list)
        counts = Counter(pts_list)
        dist_pct = {str(k2): f"{(v / len(pts_list)) * 100:.1f}%" for k2, v in sorted(counts.items())}

        print(f"    → avg_pts={avg:.2f}  optimal={orat:.2%}  dist={dist_pct}")
        results.append((avg, orat, dict(counts)))
        item_pts.extend(pts_list); item_opt.extend([int(o) for o in opt_list])
    return results, item_pts, item_opt

# ── Main Script ──────────────────────────────────────────────────────────────
with open("data/asdiv_eval_dataset.json") as f:
    rows = json.load(f)
folds = make_folds(rows)
print(f"Loaded {len(rows)} rows → 4 folds of 24\n")

print("="*55 + "\n  PHASE 1: BASELINE\n" + "="*55)
base, base_item_pts, base_item_opt = run(folds, "BASELINE")
bpts, bopt, bdist_list = zip(*base)
bdist_pct = calculate_percentages(bdist_list)


print("\n" + "="*55 + "\n  PHASE 2: PERTURBED SWEEP\n" + "="*55)
for sigma, neurons_file in NEURONS.items():
    if not os.path.exists(neurons_file):
        print(f"\n[WARNING] {neurons_file} not found. Skipping...")
        continue
        
    print(f"\n--- EVALUATING SIGMA: {sigma} ({neurons_file}) ---")
    
    with open(neurons_file) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
        
    install_hooks(neuron_map)
    modA, mod_item_pts, mod_item_opt = run(folds, f"PERTURBED ({sigma})")
    remove_hooks()

    apts, aopt, adist_list = zip(*modA)
    adist_pct = calculate_percentages(adist_list)

    t_fold_pts, p_fold_pts = stats.ttest_rel(apts, bpts)
    t_fold_opt, p_fold_opt = stats.ttest_rel(aopt, bopt)
    t_item_pts, p_item_pts = stats.ttest_rel(mod_item_pts, base_item_pts)
    t_item_opt, p_item_opt = stats.ttest_rel(mod_item_opt, base_item_opt)

    item_ttests_all_scores = {}
    for score in [10, 20, 30, 40]:
        b_binary = [1 if p == score else 0 for p in base_item_pts]
        m_binary = [1 if p == score else 0 for p in mod_item_pts]
        
        if b_binary == m_binary:
            t_val, p_val = 0.0, 1.0
        else:
            t_val, p_val = stats.ttest_rel(m_binary, b_binary)
            
        item_ttests_all_scores[f"score_{score}_item_level"] = {
            "t": float(t_val),
            "p": float(p_val),
            "n": len(base_item_pts),
            "sig": stars(p_val)
        }


    print("\n" + "="*62)
    print(f"  RESULTS FOR SIGMA: {sigma}")
    print("="*62)
    print(f"  {'':12} {'Avg pts':>12}   {'Optimal rate':>14}")
    print(f"  {'Baseline':12} {np.mean(bpts):>6.2f} +/- {np.std(bpts):.2f}   {np.mean(bopt):>8.2%} +/- {np.std(bopt):.2%}")
    print(f"  {'Perturbed':12} {np.mean(apts):>6.2f} +/- {np.std(apts):.2f}   {np.mean(aopt):>8.2%} +/- {np.std(aopt):.2%}")
    print(f"  {'Delta':12} {np.mean(apts)-np.mean(bpts):>+12.2f}   {np.mean(aopt)-np.mean(bopt):>+13.2%}")
    print("="*62)
    print("  PAIRED T-TESTS (Perturbed vs. Baseline)")
    print("="*62)
    print(f"  Avg pts   - fold-level  (n=4):  t={t_fold_pts:+.3f}  p={p_fold_pts:.4g}  {stars(p_fold_pts)}")
    print(f"  Avg pts   - item-level  (n={len(base_item_pts)}): t={t_item_pts:+.3f}  p={p_item_pts:.4g}  {stars(p_item_pts)}")
    print(f"  Optimal % - fold-level  (n=4):  t={t_fold_opt:+.3f}  p={p_fold_opt:.4g}  {stars(p_fold_opt)}")
    print(f"  Optimal % - item-level  (n={len(base_item_opt)}): t={t_item_opt:+.3f}  p={p_item_opt:.4g}  {stars(p_item_opt)}")
    
    print("\n  DISTRIBUTION T-TESTS (Item-Level):")
    for s in [10, 20, 30, 40]:
        t_data = item_ttests_all_scores[f"score_{s}_item_level"]
        print(f"  Score {s}  - item-level  (n={t_data['n']}): t={t_data['t']:+.3f}  p={t_data['p']:.4g}  {t_data['sig']}")
    print("="*62)

    os.makedirs("results", exist_ok=True)
    filename = f"results/asdiv_result.json"
    
    

    paired_tests_dict = {
        "avg_pts_fold_level":  {"t": float(t_fold_pts), "p": float(p_fold_pts), "n": 4, "sig": stars(p_fold_pts)},
        "avg_pts_item_level":  {"t": float(t_item_pts), "p": float(p_item_pts), "n": len(base_item_pts), "sig": stars(p_item_pts)},
        "optimal_fold_level":  {"t": float(t_fold_opt), "p": float(p_fold_opt), "n": 4, "sig": stars(p_fold_opt)},
        "optimal_item_level":  {"t": float(t_item_opt), "p": float(p_item_opt), "n": len(base_item_opt), "sig": stars(p_item_opt)},
    }
    paired_tests_dict.update(item_ttests_all_scores)

    with open(filename, "w") as f:
        json.dump({
            "sigma": sigma,
            "neuron_file": neurons_file,
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
            "paired_ttests": paired_tests_dict
        }, f, indent=2)
    print(f"Saved → {filename}")

print("\nPipeline Complete!")