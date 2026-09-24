import json, re, os, torch
import numpy as np
from scipy import stats
from transformers import AutoModelForVision2Seq, AutoProcessor
from collections import defaultdict, Counter

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
NEURONS_FILES = {
    "2.1": "neurons_2.1sigma.json"
}
ACTIVATIONS_DIR = "activations"
BATCH_SIZE = 8

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

if proc.tokenizer.pad_token is None:
    proc.tokenizer.pad_token = proc.tokenizer.eos_token
proc.tokenizer.padding_side = "left"

layers = model.model.language_model.layers

# ── Neutral means ────────────────────────────────────────────────────────────
parts = []
for domain in ["geo", "math"]:
    data = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu")
    parts.append(torch.stack(list(data.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

# ── Hooks Management ─────────────────────────────────────────────────────────
hooks = []


def install_hooks(neuron_map):
    remove_hooks()
    for layer_idx, neurons in neuron_map.items():
        idx = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")

        def _make(i, m):
            def _hook(_, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out

            return _hook

        hooks.append(layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    print(f"✓ Hooks ON ({sum(len(v) for v in neuron_map.values()):,} neurons)")


def remove_hooks():
    for h in hooks:
        h.remove()
    hooks.clear()


# ── Batched Inference ────────────────────────────────────────────────────────
def generate_batch(prompts: list[str]) -> list[str]:
    formatted_texts = []
    for p in prompts:
        messages = [
            {"role": "system",
             "content": "You are a participant in this experiment and must engage thoughtfully. You strictly follow rules and always output in the exact requested format."},
            {"role": "user", "content": p}
        ]
        formatted_texts.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = proc(text=formatted_texts, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    responses = []
    for i in range(len(prompts)):
        out_tokens = gen[i][inputs.input_ids.shape[1]:]
        responses.append(proc.decode(out_tokens, skip_special_tokens=True))
    return responses


# ── Evaluation ───────────────────────────────────────────────────────────────
def make_folds(rows, k=4, seed=42):
    import random;
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row["permutation"])].append(row)
    folds = [[] for _ in range(k)]
    for group in groups.values():
        rng.shuffle(group)
        for i, row in enumerate(group):
            folds[i % k].append(row)
    return folds


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def calculate_percentages(dist_list):
    total = Counter()
    for d in dist_list:
        total.update(d)
    total_items = sum(total.values())
    return {str(k): f"{(v / total_items) * 100:.1f}%" for k, v in sorted(total.items())}


def run(folds, label):
    results, item_pts, item_opt = [], [], []
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi + 1}/4 ({len(fold)} rows)")
        pts_list, opt_list = [], []

        for b_idx in range(0, len(fold), BATCH_SIZE):
            batch_rows = fold[b_idx: b_idx + BATCH_SIZE]
            prompts = [r["prompt"] for r in batch_rows]
            responses = generate_batch(prompts)

            for j, (row, resp) in enumerate(zip(batch_rows, responses)):
                cur_idx = b_idx + j
                m = re.search(r'\b([1-4])\b', resp.strip())
                choice = int(m.group(1)) if m else None
                pts = row[f"q{choice}_points"] if choice else 0
                opt = (pts == 40)
                pts_list.append(pts)
                opt_list.append(opt)
                print(
                    f"    [{cur_idx + 1:02d}/{len(fold)}] {'✓' if opt else '✗'} choice={choice} pts={pts}  {resp[:55].strip()!r}")

        avg, orat = np.mean(pts_list), np.mean(opt_list)
        counts = Counter(pts_list)
        total_items = len(pts_list)
        dist_pct = {str(k): f"{(v / total_items) * 100:.1f}%" for k, v in sorted(counts.items())}
        print(f"    → avg_pts={avg:.2f}  optimal={orat:.2%}  dist={dist_pct}")

        results.append((avg, orat, dict(counts)))
        item_pts.extend(pts_list)
        item_opt.extend([int(o) for o in opt_list])
    return results, item_pts, item_opt


# ── Load Experiment Data ─────────────────────────────────────────────────────
with open("data/asdiv_eval_dataset.json") as f:
    rows = json.load(f)
folds = make_folds(rows)
print(f"Loaded {len(rows)} rows → 4 folds of 24\n")

# ── 1) EXECUTE BASELINE ONCE ─────────────────────────────────────────────────
print("=" * 60 + "\n  BASELINE EVALUATION (RUN ONCE)\n" + "=" * 60)
base, base_item_pts, base_item_opt = run(folds, "BASELINE")
bpts, bopt, bdist_list = zip(*base)
bdist_pct = calculate_percentages(bdist_list)

os.makedirs("results", exist_ok=True)

# ── 2) EVALUATE ALL SIGMAS ───────────────────────────────────────────────────
for sigma, neurons_file in NEURONS_FILES.items():
    file_path = os.path.join(neurons_file)
    if not os.path.exists(file_path):
        print(f"Skipping {sigma}σ (file not found: {file_path})")
        continue

    print("\n" + "=" * 60 + f"\n  PERTURBED MODEL: {sigma}σ\n" + "=" * 60)
    with open(file_path) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}

    install_hooks(neuron_map)
    modA, mod_item_pts, mod_item_opt = run(folds, f"PERTURBED {sigma}σ")
    remove_hooks()

    apts, aopt, adist_list = zip(*modA)
    adist_pct = calculate_percentages(adist_list)

    # Paired Significance Tests (Fold and Item Level)
    t_fold_pts, p_fold_pts = stats.ttest_rel(apts, bpts)
    t_fold_opt, p_fold_opt = stats.ttest_rel(aopt, bopt)
    t_item_pts, p_item_pts = stats.ttest_rel(mod_item_pts, base_item_pts)
    t_item_opt, p_item_opt = stats.ttest_rel(mod_item_opt, base_item_opt)

    # =========================================================
    # =========================================================
    item_ttests_all_scores = {}
    for score in [10, 20, 30, 40]:
        b_binary = [1 if p == score else 0 for p in base_item_pts]
        m_binary = [1 if p == score else 0 for p in mod_item_pts]

        if b_binary == m_binary:
            t_val, p_val = 0.0, 1.0
        else:
            t_val, p_val = stats.ttest_rel(m_binary, b_binary)

        item_ttests_all_scores[f"score_{score}_item_level"] = {
            "t": float(t_val) if not np.isnan(t_val) else 0.0,
            "p": float(p_val) if not np.isnan(p_val) else 1.0,
            "n": len(base_item_pts),
            "sig": stars(p_val if not np.isnan(p_val) else 1.0)
        }

    print("\n" + "-" * 60)
    print(f"  RESULTS FOR {sigma}σ  (Baseline vs Perturbed)")
    print("-" * 60)
    print(f"  Baseline:  Avg pts = {np.mean(bpts):.2f} ± {np.std(bpts):.2f} | Optimal = {np.mean(bopt):.2%}")
    print(f"  Perturbed: Avg pts = {np.mean(apts):.2f} ± {np.std(apts):.2f} | Optimal = {np.mean(aopt):.2%}")
    print(f"  Delta pts: {np.mean(apts) - np.mean(bpts):+.2f}")
    print(f"  Item-level pts p-val: {p_item_pts:.4g} ({stars(p_item_pts)})")
    print(f"  Item-level opt p-val: {p_item_opt:.4g} ({stars(p_item_opt)})")

    print("\n  DISTRIBUTION T-TESTS (Item-Level):")
    for s in [10, 20, 30, 40]:
        t_data = item_ttests_all_scores[f"score_{s}_item_level"]
        print(
            f"  Score {s}  — item-level  (n={t_data['n']}): t={t_data['t']:+.3f}  p={t_data['p']:.4g}  {t_data['sig']}")

    # =========================================================
    # =========================================================
    out_path = f"results/asdiv_results.json"

    paired_tests_dict = {
        "avg_pts_fold_level": {"t": float(t_fold_pts) if not np.isnan(t_fold_pts) else 0.0,
                               "p": float(p_fold_pts) if not np.isnan(p_fold_pts) else 1.0, "n": 4,
                               "sig": stars(p_fold_pts)},
        "avg_pts_item_level": {"t": float(t_item_pts) if not np.isnan(t_item_pts) else 0.0,
                               "p": float(p_item_pts) if not np.isnan(p_item_pts) else 1.0, "n": len(base_item_pts),
                               "sig": stars(p_item_pts)},
        "optimal_fold_level": {"t": float(t_fold_opt) if not np.isnan(t_fold_opt) else 0.0,
                               "p": float(p_fold_opt) if not np.isnan(p_fold_opt) else 1.0, "n": 4,
                               "sig": stars(p_fold_opt)},
        "optimal_item_level": {"t": float(t_item_opt) if not np.isnan(t_item_opt) else 0.0,
                               "p": float(p_item_opt) if not np.isnan(p_item_opt) else 1.0, "n": len(base_item_opt),
                               "sig": stars(p_item_opt)},
    }
    paired_tests_dict.update(item_ttests_all_scores)

    with open(out_path, "w") as f:
        json.dump({
            "sigma": sigma,
            "baseline": base,
            "perturbed": modA,
            "point_distributions_percentage": {
                "baseline": bdist_pct,
                "perturbed": adist_pct
            },
            "summary": {
                "baseline_pts": f"{np.mean(bpts):.2f}±{np.std(bpts):.2f}",
                "perturbed_pts": f"{np.mean(apts):.2f}±{np.std(apts):.2f}",
                "delta_pts": f"{np.mean(apts) - np.mean(bpts):+.2f}"
            },
            "paired_ttests": paired_tests_dict
        }, f, indent=2)
    print(f"Saved → {out_path}")