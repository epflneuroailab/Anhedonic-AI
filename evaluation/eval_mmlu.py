"""
eval_model_a_mmlu_difficulty.py
================================================================================
Evaluates baseline vs Model A on 10 randomly chosen subjects from
data/mmlu_eval_difficulty/, using the same hook/inference pattern as
asdiv version.

Eval structure per subject:
  - 96 rows split into 5 subsets (~19-20 rows each) used as folds
  - "optimal" = model chose the question with 40 pts (the hardest one)
  - Metrics: avg_pts, optimal_rate  (per fold, per subject, and global)

Output: results/eval_mmlu_difficulty.json
"""

import json, re, os, random, torch
import numpy as np
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from collections import defaultdict

MODEL_PATH      = "../../models/qwen2-vl-7b"
NEURONS_JSON    = "neurons_A.json"
ACTIVATIONS_DIR = "../extraction/activations"
MMLU_DIR        = "data/mmlu_eval_difficulty"
N_SUBJECTS      = 10
SEED            = 42

random.seed(SEED)

# ── Pick 10 random subjects ───────────────────────────────────────────────────
all_files = sorted(f for f in os.listdir(MMLU_DIR) if f.endswith(".csv"))
chosen    = random.sample(all_files, N_SUBJECTS)
print(f"Selected {N_SUBJECTS} subjects:")
for f in chosen:
    print(f"  • {f.replace('.csv','')}")

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
proc   = AutoProcessor.from_pretrained(MODEL_PATH)
layers = model.model.language_model.layers

# ── Neutral means + neuron map ────────────────────────────────────────────────
parts = []
for domain in ["geo", "math"]:
    data = torch.load(
        os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"),
        map_location="cpu"
    )
    parts.append(torch.stack(list(data.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

with open(NEURONS_JSON) as f:
    neuron_map = {int(k): v for k, v in json.load(f).items()}

# ── Hooks ─────────────────────────────────────────────────────────────────────
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

# ── Inference ─────────────────────────────────────────────────────────────────
def generate(prompt):
    text   = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return proc.batch_decode(
        [gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True
    )[0]

# ── Load one subject CSV into fold lists ──────────────────────────────────────
def load_subject(csv_path: str) -> list[list[dict]]:
    """
    Returns a list of 5 folds (one per Subset value 1-5).
    Each fold is a list of row dicts with keys:
        prompt, reward_1..4  (int), subset
    """
    df    = pd.read_csv(csv_path)
    folds = defaultdict(list)
    for _, row in df.iterrows():
        folds[int(row["Subset"])].append({
            "prompt":   row["Full_Prompt"],
            "reward_1": int(row["Reward_1"]),
            "reward_2": int(row["Reward_2"]),
            "reward_3": int(row["Reward_3"]),
            "reward_4": int(row["Reward_4"]),
            "subset":   int(row["Subset"]),
        })
    # Return folds in order 1-5
    return [folds[s] for s in sorted(folds.keys())]

# ── Eval one subject, one condition ───────────────────────────────────────────
def run_subject(folds: list[list[dict]], subject: str, label: str) -> list[tuple]:
    """
    Returns list of (avg_pts, optimal_rate) per fold.
    """
    results = []
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] {subject}  Fold {fi+1}/5 ({len(fold)} rows)")
        pts_list, opt_list = [], []
        for i, row in enumerate(fold):
            resp   = generate(row["prompt"])
            m      = re.search(r'\b([1-4])\b', resp.strip())
            choice = int(m.group(1)) if m else None
            pts    = row[f"reward_{choice}"] if choice else 0
            opt    = (pts == 40)
            pts_list.append(pts)
            opt_list.append(opt)
            print(f"    [{i+1:02d}/{len(fold):02d}] {'✓' if opt else '✗'} "
                  f"choice={choice} pts={pts}  {resp[:55].strip()!r}")
        avg, orat = np.mean(pts_list), np.mean(opt_list)
        print(f"    → avg_pts={avg:.2f}  optimal={orat:.2%}")
        results.append((float(avg), float(orat)))
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
all_results = {}   # subject → {"baseline": [...], "model_a": [...]}

for fname in chosen:
    subject  = fname.replace(".csv", "")
    csv_path = os.path.join(MMLU_DIR, fname)
    folds    = load_subject(csv_path)

    print("\n" + "="*60)
    print(f"  SUBJECT: {subject}")
    print("="*60)

    print(f"\n--- BASELINE ---")
    base = run_subject(folds, subject, "BASELINE")

    print(f"\n--- MODEL A ---")
    install_hooks()
    modA = run_subject(folds, subject, "MODEL A")
    remove_hooks()

    all_results[subject] = {"baseline": base, "model_a": modA}

# ── Per-subject summary ───────────────────────────────────────────────────────
print("\n\n" + "="*72)
print("  PER-SUBJECT SUMMARY  (5 folds each)")
print("="*72)
print(f"  {'Subject':<35} {'Base pts':>9}  {'ModA pts':>9}  {'Δ pts':>7}  {'Base opt':>9}  {'ModA opt':>9}  {'Δ opt':>7}")
print(f"  {'─'*35} {'─'*9}  {'─'*9}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*7}")

global_base_pts, global_moda_pts   = [], []
global_base_opt, global_moda_opt   = [], []

for subject, res in all_results.items():
    b_pts = [r[0] for r in res["baseline"]]
    b_opt = [r[1] for r in res["baseline"]]
    a_pts = [r[0] for r in res["model_a"]]
    a_opt = [r[1] for r in res["model_a"]]

    global_base_pts.extend(b_pts); global_moda_pts.extend(a_pts)
    global_base_opt.extend(b_opt); global_moda_opt.extend(a_opt)

    d_pts = np.mean(a_pts) - np.mean(b_pts)
    d_opt = np.mean(a_opt) - np.mean(b_opt)
    print(f"  {subject:<35} {np.mean(b_pts):>6.2f}±{np.std(b_pts):.2f}  "
          f"{np.mean(a_pts):>6.2f}±{np.std(a_pts):.2f}  "
          f"{d_pts:>+7.2f}  "
          f"{np.mean(b_opt):>7.2%}±{np.std(b_opt):.2%}  "
          f"{np.mean(a_opt):>7.2%}±{np.std(a_opt):.2%}  "
          f"{d_opt:>+7.2%}")

# ── Global summary ────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  GLOBAL SUMMARY  (across all 10 subjects × 5 folds)")
print("="*72)
print(f"  {'':12} {'Avg pts':>14}   {'Optimal rate':>16}")
print(f"  {'─'*12} {'─'*14}   {'─'*16}")
print(f"  {'Baseline':12} {np.mean(global_base_pts):>6.2f} ± {np.std(global_base_pts):.2f}   "
      f"{np.mean(global_base_opt):>8.2%} ± {np.std(global_base_opt):.2%}")
print(f"  {'Model A':12} {np.mean(global_moda_pts):>6.2f} ± {np.std(global_moda_pts):.2f}   "
      f"{np.mean(global_moda_opt):>8.2%} ± {np.std(global_moda_opt):.2%}")
print(f"  {'Δ':12} {np.mean(global_moda_pts)-np.mean(global_base_pts):>+14.2f}   "
      f"{np.mean(global_moda_opt)-np.mean(global_base_opt):>+15.2%}")
print("="*72)

# ── Save results ──────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
out = {
    "subjects_evaluated": [f.replace(".csv","") for f in chosen],
    "seed": SEED,
    "per_subject": {
        subj: {
            "baseline": res["baseline"],
            "model_a":  res["model_a"],
            "summary": {
                "baseline_pts": f"{np.mean([r[0] for r in res['baseline']]):.2f}"
                                f"±{np.std([r[0] for r in res['baseline']]):.2f}",
                "modelA_pts":   f"{np.mean([r[0] for r in res['model_a']]):.2f}"
                                f"±{np.std([r[0] for r in res['model_a']]):.2f}",
                "delta_pts":    f"{np.mean([r[0] for r in res['model_a']])-np.mean([r[0] for r in res['baseline']]):+.2f}",
                "baseline_opt": f"{np.mean([r[1] for r in res['baseline']]):.2%}",
                "modelA_opt":   f"{np.mean([r[1] for r in res['model_a']]):.2%}",
                "delta_opt":    f"{np.mean([r[1] for r in res['model_a']])-np.mean([r[1] for r in res['baseline']]):+.2%}",
            }
        }
        for subj, res in all_results.items()
    },
    "global_summary": {
        "baseline_pts": f"{np.mean(global_base_pts):.2f}±{np.std(global_base_pts):.2f}",
        "modelA_pts":   f"{np.mean(global_moda_pts):.2f}±{np.std(global_moda_pts):.2f}",
        "delta_pts":    f"{np.mean(global_moda_pts)-np.mean(global_base_pts):+.2f}",
        "baseline_opt": f"{np.mean(global_base_opt):.2%}",
        "modelA_opt":   f"{np.mean(global_moda_opt):.2%}",
        "delta_opt":    f"{np.mean(global_moda_opt)-np.mean(global_base_opt):+.2%}",
    }
}
with open("results/eval_mmlu_difficulty.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved → results/eval_mmlu_difficulty.json")