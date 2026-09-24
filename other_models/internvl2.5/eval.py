import json, re, os, sys, time, random
import numpy as np
from scipy import stats
from collections import defaultdict, Counter
import torch
from transformers import AutoTokenizer, AutoModel, GenerationMixin, GenerationConfig

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("MODEL_PATH", "OpenGVLab/InternVL2_5-8B")
ACTIVATIONS_DIR = "activations"
DATASET_PATH    = "data/asdiv_eval_dataset.json"


NEURONS_JSON    = os.environ.get("NEURONS_JSON", "neurons.json")
RESULTS_PATH    = os.environ.get("RESULTS_PATH", "results/asdiv_results.json")

MIN_LAYER       = int(os.environ.get("MIN_LAYER", 18))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", 192))
MAX_NEW_TOKENS  = 512
CHECK_EVERY     = int(os.environ.get("CHECK_EVERY", 16))
K_FOLDS, SEED   = 4, 42
DEVICE          = "cuda"

PARSER      = os.environ.get("PARSER", "anchored")     
EARLY_STOP  = os.environ.get("EARLY_STOP", "1") == "1"
VERIFY_N    = int(os.environ.get("VERIFY_N", 4))
VERIFY_MAX_MISMATCH = int(os.environ.get("VERIFY_MAX_MISMATCH", 1))
SAVE_RAW    = os.environ.get("SAVE_RAW", "1") == "1"   

SYSTEM_PROMPT = ("You are a participant in this experiment and must engage thoughtfully. "
                 "You strictly follow rules and always output in the exact requested format.")

# ── Parser ───────────────────────────────────────────────────────────────────
BARE_RE    = re.compile(r'\b([1-4])\b')
CHOSEN_RE  = re.compile(r'Chosen\s*Question\s*[:\-]?\s*\[?\s*(?:Question\s*)?([1-4])', re.I)
ANY_CHOSEN = re.compile(r'Chosen\s*Question', re.I)

def extract_choice(resp, parser=None):
    parser = parser or PARSER
    if not resp:
        return None
    if parser == "bare":
        m = BARE_RE.search(resp.strip())
        return int(m.group(1)) if m else None
    m = CHOSEN_RE.search(resp)
    return int(m.group(1)) if m else None

def can_stop(text):
    s = text.strip()
    if not s:
        return False
    rx = BARE_RE if PARSER == "bare" else CHOSEN_RE
    m = rx.search(s)
    return m is not None and m.end() < len(s)

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModel.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE,
    trust_remote_code=True, use_flash_attn=True,
).eval()

if not hasattr(model.language_model, "generate"):
    model.language_model.__class__.__bases__ = (GenerationMixin,) + model.language_model.__class__.__bases__
if getattr(model.language_model, "generation_config", None) is None:
    model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)

layers  = model.language_model.model.layers
LM      = model.language_model.model
HEAD    = getattr(model.language_model, "output", None) or model.language_model.lm_head
_eos    = model.language_model.generation_config.eos_token_id or tokenizer.eos_token_id
EOS_IDS = torch.tensor(_eos if isinstance(_eos, (list, tuple)) else [_eos], device=DEVICE)

try:
    import flash_attn  # noqa: F401
    print(f"✓ flash-attn importable")
except ImportError:
    print("✗ flash-attn NOT importable in this environment.")

# ── Neutral means ────────────────────────────────────────────────────────────
parts = []
for domain in ["geo", "math"]:
    data = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu")
    parts.append(torch.stack(list(data.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

def get_ffn_module(layer):
    if hasattr(layer, "feed_forward"): return layer.feed_forward
    if hasattr(layer, "mlp"):          return layer.mlp
    raise AttributeError("FFN module not found.")

# ── Hooks ────────────────────────────────────────────────────────────────────
hooks = []

def install_hooks(neuron_map):
    assert not hooks, "hooks already installed"
    n_total = 0
    for layer_idx, neurons in neuron_map.items():
        l = int(layer_idx)
        if l < MIN_LAYER or not neurons:
            continue
        idx   = torch.tensor(neurons).long().to(DEVICE)
        means = torch.tensor(mean_acts[l, neurons], dtype=torch.bfloat16).to(DEVICE)
        def _make(i, m):
            def _hook(_, _in, out):
                if out.dim() == 2: out[:, i]    = m.unsqueeze(0)
                else:              out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        hooks.append(get_ffn_module(layers[l]).act_fn.register_forward_hook(_make(idx, means)))
        n_total += len(neurons)
    print(f"✓ Hooks ON  ({n_total:,} neurons)")
    return n_total

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF")

# ── Cached greedy decoding ───────────────────────────────────────────────────
def build_text(prompt):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": prompt}],
        tokenize=False, add_generation_prompt=True)

@torch.inference_mode()
def greedy_cached(texts, max_new=MAX_NEW_TOKENS, early_stop=None):
    early_stop = EARLY_STOP if early_stop is None else early_stop
    enc  = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    ids, prompt_mask = enc.input_ids, enc.attention_mask
    B, prompt_len = ids.shape
    pos  = (prompt_mask.cumsum(-1) - 1).clamp(min=0)

    mask_full = torch.zeros(B, prompt_len + max_new, dtype=prompt_mask.dtype, device=DEVICE)
    mask_full[:, :prompt_len] = prompt_mask

    past, cur = None, ids
    toks  = torch.full((B, max_new), tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
    done  = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    n_gen = torch.zeros(B, dtype=torch.long, device=DEVICE)

    for step in range(max_new):
        cur_len = prompt_len + step
        out  = LM(input_ids=cur, attention_mask=mask_full[:, :cur_len], position_ids=pos,
                  past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt  = HEAD(out.last_hidden_state[:, -1, :]).argmax(-1)
        nxt  = torch.where(done, torch.full_like(nxt, tokenizer.pad_token_id), nxt)
        toks[:, step] = nxt
        n_gen += (~done).long()
        done  |= torch.isin(nxt, EOS_IDS)

        if early_stop and (step + 1) % CHECK_EVERY == 0 and not bool(done.all()):
            active = (~done).nonzero(as_tuple=True)[0]
            if active.numel() > 0:
                decoded = tokenizer.batch_decode(toks[active, :step + 1], skip_special_tokens=True)
                for idx, t in zip(active.tolist(), decoded):
                    if can_stop(t):
                        done[idx] = True
        if bool(done.all()):
            break

        cur = nxt[:, None]
        mask_full[:, cur_len] = 1
        pos = pos[:, -1:] + 1

    return [tokenizer.decode(toks[i, :n_gen[i]], skip_special_tokens=True) for i in range(B)]

@torch.inference_mode()
def generate_uncached(texts, max_new=MAX_NEW_TOKENS):
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    gen = model.language_model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                        use_cache=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

def verify(texts):
    print(f"\n[verify] cached+early-stop vs uncached, {len(texts)} prompts...")
    slow, fast = generate_uncached(texts, 192), greedy_cached(texts, 192)
    bad = 0
    for i, (s, f) in enumerate(zip(slow, fast)):
        cs, cf = extract_choice(s), extract_choice(f)
        if cs != cf:
            bad += 1
            print(f"  ✗ [{i}] uncached={cs} cached={cf}\n      {s[:110]!r}\n      {f[:110]!r}")
        else:
            print(f"  ✓ [{i}] choice={cs}  len {len(s)}→{len(f)}")
    if bad > VERIFY_MAX_MISMATCH:
        raise RuntimeError("Cached path disagrees with the uncached baseline beyond tolerance.")
    else:
        print("[verify] OK\n")

# ── Folds ────────────────────────────────────────────────────────────────────
def make_folds(rows, k=K_FOLDS, seed=SEED):
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

# ── Eval one pass ────────────────────────────────────────────────────────────
def run(items, lens, fold_indices, label):
    t0 = time.time()
    n = len(items)
    pts  = np.zeros(n); opt  = np.zeros(n, dtype=int)
    bpts = np.zeros(n); bopt = np.zeros(n, dtype=int)
    apts = np.zeros(n); aopt = np.zeros(n, dtype=int)
    nofind = np.zeros(n, dtype=int); refuse = np.zeros(n, dtype=int)
    disagree = np.zeros(n, dtype=int)
    raw = [None] * n

    fold_results = []
    for fi, idxs in enumerate(fold_indices):
        print(f"\n  [{label}] Fold {fi+1}/{len(fold_indices)} ({len(idxs)} rows)")
        order = sorted(idxs, key=lambda i: lens[i])   

        for s in range(0, len(order), BATCH_SIZE):
            chunk = order[s:s + BATCH_SIZE]
            resps = greedy_cached([items[i]["_text"] for i in chunk])
            for i, resp in zip(chunk, resps):
                c_a = extract_choice(resp, "anchored")
                c_b = extract_choice(resp, "bare")
                choice = c_a if PARSER == "anchored" else c_b

                apts[i] = items[i][f"q{c_a}_points"] if c_a else 0
                bpts[i] = items[i][f"q{c_b}_points"] if c_b else 0
                aopt[i] = int(apts[i] == 40); bopt[i] = int(bpts[i] == 40)

                pts[i] = apts[i] if PARSER == "anchored" else bpts[i]
                opt[i] = int(pts[i] == 40)

                nofind[i]   = int(choice is None)
                refuse[i]   = int(c_a is None and not ANY_CHOSEN.search(resp))
                disagree[i] = int(c_a != c_b)
                raw[i]      = resp
            print(f"    [{label}] fold {fi+1}: {min(s + BATCH_SIZE, len(order))}/{len(order)} "
                  f"generated ({time.time()-t0:.0f}s)")

        fold_pts, fold_opt = pts[idxs], opt[idxs]
        counts = Counter(fold_pts.tolist())
        dist = {str(int(k)): f"{v/len(idxs)*100:.1f}%" for k, v in sorted(counts.items())}
        avg_pts, opt_rate = float(fold_pts.mean()), float(fold_opt.mean())
        print(f"    [{label}] Fold {fi+1} → avg_pts={avg_pts:.2f}  optimal={opt_rate:.2%}  dist={dist}")
        fold_results.append((avg_pts, opt_rate, {int(k): int(v) for k, v in sorted(counts.items())}))

    print(f"\n  [{label}] total {time.time()-t0:.0f}s  |  unparsed {nofind.sum()}/{n} "
          f"(no choice line: {refuse.sum()}, parser miss: {nofind.sum()-refuse.sum()})")
    return {"folds": fold_results, "pts": pts, "opt": opt, "nofind": nofind, "refuse": refuse,
            "raw": raw, "bare_pts": bpts, "bare_opt": bopt,
            "anch_pts": apts, "anch_opt": aopt, "disagree": disagree}

def parser_block(R, label):
    d = R["disagree"].sum(); n = len(R["pts"])
    print(f"  PARSER COMPARISON ({label})")
    print(f"    disagreements            : {d}/{n}")
    print(f"    avg pts   bare={R['bare_pts'].mean():6.2f}  anchored={R['anch_pts'].mean():6.2f}  "
          f"(Δ {R['anch_pts'].mean()-R['bare_pts'].mean():+.2f})")
    return {"disagreements": int(d),
            "bare_pts": float(R["bare_pts"].mean()), "anchored_pts": float(R["anch_pts"].mean()),
            "bare_opt_rate": float(R["bare_opt"].mean()), "anchored_opt_rate": float(R["anch_opt"].mean()),
            "unparsed": int(R["nofind"].sum()), "no_choice_line": int(R["refuse"].sum()),
            "parser_miss": int(R["nofind"].sum() - R["refuse"].sum())}

def calculate_percentages(dist_list):
    total = Counter()
    for d in dist_list:
        total.update(d)
    tot = sum(total.values())
    return {str(k): f"{(v / tot) * 100:.1f}%" for k, v in sorted(total.items())}

def ttest(mod, base):
    t, p = stats.ttest_rel(mod, base)
    return (float(t) if not np.isnan(t) else 0.0,
            float(p) if not np.isnan(p) else 1.0)

def detailed(items, raw, pts, opt):
    out = []
    for i, it in enumerate(items):
        d = {"prompt": it["prompt"], "raw_response": raw[i],
             "parsed_choice": extract_choice(raw[i]),
             "points_awarded": int(pts[i]), "optimal": bool(opt[i])}
        for q in (1, 2, 3, 4):
            d[f"q{q}_points"] = it[f"q{q}_points"]      
        out.append(d)
    return out

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(NEURONS_JSON):
        raise FileNotFoundError(f"Neuron file not found: {NEURONS_JSON}")

    with open(NEURONS_JSON) as f:
        raw_neuron_map = json.load(f)
        neuron_map = {int(k): v for k, v in raw_neuron_map.items()}

    print(f"Loading single neuron map from: {NEURONS_JSON}")

    rows    = json.load(open(DATASET_PATH))
    folds   = make_folds(rows)
    items   = [row for fold in folds for row in fold]
    for r in items:
        r["_text"] = build_text(r["prompt"])
    lens    = [len(tokenizer(r["_text"]).input_ids) for r in items]

    fold_bounds  = np.cumsum([0] + [len(f) for f in folds])
    fold_indices = [list(range(int(fold_bounds[fi]), int(fold_bounds[fi + 1]))) for fi in range(K_FOLDS)]

    print(f"Loaded {len(rows)} rows → {K_FOLDS} folds of {len(folds[0])}  |  prompt tokens "
          f"{min(lens)}–{max(lens)}  |  batch {BATCH_SIZE}  |  parser={PARSER}  "
          f"|  early_stop={EARLY_STOP}  |  MIN_LAYER={MIN_LAYER}  |  verify_tol={VERIFY_MAX_MISMATCH}\n")

    if VERIFY_N:
        verify_idxs = sorted(fold_indices[0], key=lambda i: lens[i])[:VERIFY_N]
        verify([items[i]["_text"] for i in verify_idxs])

    print("=" * 55 + "\n  BASELINE\n" + "=" * 55)
    B = run(items, lens, fold_indices, "BASELINE")
    b_pts, b_opt, b_nf = B["pts"], B["opt"], B["nofind"]
    bpts, bopt, bdist_list = zip(*B["folds"])
    bdist_pct = calculate_percentages(bdist_list)
    print("=" * 62)
    b_parser = parser_block(B, "BASELINE")
    print("=" * 62)

    print("\n" + "=" * 62 + f"\n  PERTURBED MODEL\n" + "=" * 62)
    n_neurons = install_hooks(neuron_map)
    M = run(items, lens, fold_indices, f"PERTURBED")
    remove_hooks()
    
    m_pts, m_opt, m_nf, m_rf = M["pts"], M["opt"], M["nofind"], M["refuse"]
    apts, aopt, adist_list = zip(*M["folds"])
    adist_pct = calculate_percentages(adist_list)

    t_fold_pts, p_fold_pts = ttest(apts, bpts)
    t_fold_opt, p_fold_opt = ttest(aopt, bopt)
    t_item_pts, p_item_pts = ttest(m_pts, b_pts)
    t_item_opt, p_item_opt = ttest(m_opt, b_opt)
    both = (b_nf == 0) & (m_nf == 0)
    t_par_pts, p_par_pts = ttest(m_pts[both], b_pts[both])
    t_bare_pts, p_bare_pts = ttest(M["bare_pts"], B["bare_pts"])


    item_ttests_all_scores = {}
    for score in [10, 20, 30, 40]:
        b_binary = [1 if p == score else 0 for p in b_pts]
        m_binary = [1 if p == score else 0 for p in m_pts]
        if np.array_equal(b_binary, m_binary):
            t_val, p_val = 0.0, 1.0
        else:
            t_val, p_val = stats.ttest_rel(m_binary, b_binary)
        
        item_ttests_all_scores[f"score_{score}_item_level"] = {
            "t": float(t_val) if not np.isnan(t_val) else 0.0,
            "p": float(p_val) if not np.isnan(p_val) else 1.0,
            "n": len(b_pts),
            "sig": stars(float(p_val) if not np.isnan(p_val) else 1.0)
        }

    print("\n" + "=" * 62)
    print(f"  RESULTS  ({K_FOLDS} folds × {len(folds[0])} rows, {n_neurons:,} neurons)")
    print("=" * 62)
    print(f"  {'':12} {'Avg pts':>12}   {'Optimal rate':>14}")
    print(f"  {'-'*12} {'-'*12}   {'-'*14}")
    print(f"  {'Baseline':12} {np.mean(bpts):>6.2f} ± {np.std(bpts):.2f}   {np.mean(bopt):>8.2%} ± {np.std(bopt):.2%}")
    print(f"  {'Perturbed':12} {np.mean(apts):>6.2f} ± {np.std(apts):.2f}   {np.mean(aopt):>8.2%} ± {np.std(aopt):.2%}")
    print(f"  {'D':12} {np.mean(apts)-np.mean(bpts):>+12.2f}   {np.mean(aopt)-np.mean(bopt):>+13.2%}")
    print("=" * 62)
    print("  PAIRED T-TESTS (perturbed vs. baseline)")
    print("=" * 62)
    print(f"  Avg pts   - fold-level  (n={K_FOLDS}):  t={t_fold_pts:+.3f}  p={p_fold_pts:.4g}  {stars(p_fold_pts)}")
    print(f"  Avg pts   - item-level  (n={len(b_pts)}): t={t_item_pts:+.3f}  p={p_item_pts:.4g}  {stars(p_item_pts)}")
    print(f"  Optimal % - fold-level  (n={K_FOLDS}):  t={t_fold_opt:+.3f}  p={p_fold_opt:.4g}  {stars(p_fold_opt)}")
    print(f"  Optimal % - item-level  (n={len(b_opt)}): t={t_item_opt:+.3f}  p={p_item_opt:.4g}  {stars(p_item_opt)}")
    
    print("\n  DISTRIBUTION T-TESTS (Item-Level):")
    for sc in [10, 20, 30, 40]:
        td = item_ttests_all_scores[f"score_{sc}_item_level"]
        print(f"  Score {sc}  — item-level  (n={td['n']}): t={td['t']:+.3f}  p={td['p']:.4g}  {td['sig']}")

    print("=" * 62)
    m_parser = parser_block(M, "PERTURBED")
    print("=" * 62)
    print("  POINT DISTRIBUTIONS (Percentages across all folds):")
    print(f"  Baseline:  {bdist_pct}")
    print(f"  Perturbed: {adist_pct}")
    print("=" * 62)

    paired_tests_dict = {
        "avg_pts_fold_level": {"t": t_fold_pts, "p": p_fold_pts, "n": K_FOLDS,    "sig": stars(p_fold_pts)},
        "avg_pts_item_level": {"t": t_item_pts, "p": p_item_pts, "n": len(b_pts), "sig": stars(p_item_pts)},
        "optimal_fold_level": {"t": t_fold_opt, "p": p_fold_opt, "n": K_FOLDS,    "sig": stars(p_fold_opt)},
        "optimal_item_level": {"t": t_item_opt, "p": p_item_opt, "n": len(b_opt), "sig": stars(p_item_opt)},
    }
    paired_tests_dict.update(item_ttests_all_scores)

    final = {
        "config": {"model": MODEL_PATH, "parser": PARSER, "min_layer": MIN_LAYER,
                   "early_stop": EARLY_STOP, "batch_size": BATCH_SIZE,
                   "source_file": NEURONS_JSON, "dataset": DATASET_PATH},
        "n_neurons_patched": n_neurons,
        "baseline": B["folds"],
        "perturbed": M["folds"],
        "point_distributions_percentage": {"baseline": bdist_pct, "perturbed": adist_pct},
        "summary": {
            "baseline_pts":  f"{np.mean(bpts):.2f}±{np.std(bpts):.2f}",
            "perturbed_pts": f"{np.mean(apts):.2f}±{np.std(apts):.2f}",
            "delta_pts":     f"{np.mean(apts)-np.mean(bpts):+.2f}",
        },
        "paired_ttests": paired_tests_dict,
        "robustness": {
            "avg_pts_parsed_in_both": {"t": t_par_pts, "p": p_par_pts, "n": int(both.sum()), "sig": stars(p_par_pts)},
            "avg_pts_bare_parser":    {"t": t_bare_pts, "p": p_bare_pts, "n": len(b_pts),    "sig": stars(p_bare_pts),
                                       "delta_pts": float(M["bare_pts"].mean() - B["bare_pts"].mean())},
        },
        "parser_comparison_baseline": b_parser,
        "parser_comparison_perturbed": m_parser,
    }
    if SAVE_RAW:
        final["detailed_responses"] = detailed(items, M["raw"], m_pts, m_opt)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nExecution finished successfully. Saved -> {RESULTS_PATH}")

if __name__ == "__main__":
    main()