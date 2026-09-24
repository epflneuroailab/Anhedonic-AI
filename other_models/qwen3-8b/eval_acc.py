import json, re, os, torch
import numpy as np
from scipy import stats
from transformers import AutoModelForVision2Seq, AutoProcessor
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
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
        if not neurons: continue
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
            max_new_tokens=64,
            do_sample=False
        )

    responses = []
    for i in range(len(prompts)):
        out_tokens = gen[i][inputs.input_ids.shape[1]:]
        responses.append(proc.decode(out_tokens, skip_special_tokens=True))
    return responses


# ── Scoring ──────────────────────────────────────────────────────────────────
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
    ref = extract_number(str(gold))
    if pred is None or ref is None:
        return False
    if ref == 0:
        return abs(pred) < 0.01
    return abs(pred - ref) / max(abs(ref), 1e-9) < 0.01


# ── Folds & Utils ────────────────────────────────────────────────────────────
def make_folds(rows, k=4, seed=42):
    import random;
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get("source_permutation", row.get("permutation", [])))].append(row)
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


def run(folds, label):
    fold_stats = []
    all_rows = []

    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi + 1}/4 ({len(fold)} rows)")
        correct_list = []

        for b_idx in range(0, len(fold), BATCH_SIZE):
            batch_rows = fold[b_idx: b_idx + BATCH_SIZE]
            prompts = [r["prompt"] for r in batch_rows]
            responses = generate_batch(prompts)

            for j, (row, resp) in enumerate(zip(batch_rows, responses)):
                correct = is_correct(resp, row["answer"])
                correct_list.append(correct)
                status = "✓" if correct else "✗"

                all_rows.append({
                    **row,
                    "label": label,
                    "fold": fi + 1,
                    "response": resp,
                    "correct": correct
                })

        acc = np.mean(correct_list)
        print(f"    → Accuracy = {acc:.2%}")
        fold_stats.append({"fold": fi + 1, "accuracy": float(acc)})

    return fold_stats, all_rows


# ── Load Dataset ─────────────────────────────────────────────────────────────
with open("data/asdiv_accuracy_dataset.json") as f:
    rows = json.load(f)
folds = make_folds(rows)
print(f"Loaded {len(rows)} rows → 4 folds")

# ── 1) EXECUTE BASELINE ONCE ─────────────────────────────────────────────────
print("=" * 60 + "\n  BASELINE ACCURACY (RUN ONCE)\n" + "=" * 60)
base_stats, base_rows = run(folds, "BASELINE")
b_fold_acc = [s["accuracy"] for s in base_stats]
b_item_acc = [1 if r["correct"] else 0 for r in base_rows]

os.makedirs("results", exist_ok=True)

# ── 2) EVALUATE ALL SIGMAS ───────────────────────────────────────────────────
for sigma, neurons_file in NEURONS_FILES.items():
    file_path = os.path.join(neurons_file)
    if not os.path.exists(file_path):
        print(f"Skipping {sigma}σ (file not found: {file_path})")
        continue

    print("\n" + "=" * 60 + f"\n  PERTURBED ACCURACY: {sigma}σ\n" + "=" * 60)
    with open(file_path) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}

    install_hooks(neuron_map)
    pert_stats, pert_rows = run(folds, f"PERTURBED {sigma}σ")
    remove_hooks()

    p_fold_acc = [s["accuracy"] for s in pert_stats]
    p_item_acc = [1 if r["correct"] else 0 for r in pert_rows]

    # Paired Significance Tests
    if b_fold_acc == p_fold_acc:
        t_fold, p_fold = 0.0, 1.0
    else:
        t_fold, p_fold = stats.ttest_rel(p_fold_acc, b_fold_acc)

    if b_item_acc == p_item_acc:
        t_item, p_item = 0.0, 1.0
    else:
        t_item, p_item = stats.ttest_rel(p_item_acc, b_item_acc)

    bm = np.mean(b_fold_acc)
    bs = np.std(b_fold_acc) / 2
    pm = np.mean(p_fold_acc)
    ps = np.std(p_fold_acc) / 2

    print("\n" + "-" * 60)
    print(f"  RESULTS FOR {sigma}σ  (Baseline vs Perturbed Accuracy)")
    print("-" * 60)
    print(f"  Baseline  : {bm:.2%} ± {bs:.2%}")
    print(f"  Perturbed : {pm:.2%} ± {ps:.2%}")
    print(f"  Delta     : {pm - bm:+.2%}")
    print(f"  Fold-level p-val: {p_fold:.4g} ({stars(p_fold)})")
    print(f"  Item-level p-val: {p_item:.4g} ({stars(p_item)})")

    # Save to JSON
    out_path = f"results/accuracy_{sigma}.json"
    with open(out_path, "w") as f:
        json.dump({
            "sigma": sigma,
            "baseline": {"folds": base_stats, "rows": base_rows},
            "perturbed": {"folds": pert_stats, "rows": pert_rows},
            "statistics": {
                "delta_accuracy": float(pm - bm),
                "t_test_fold": {"t": float(t_fold), "p": float(p_fold), "sig": stars(p_fold)},
                "t_test_item": {"t": float(t_item), "p": float(p_item), "sig": stars(p_item)}
            }
        }, f, indent=2)
    print(f"Saved → {out_path}")

print("\nPipeline Complete!")