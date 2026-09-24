import json
import argparse
import pandas as pd


ABS_NEURONS_FILE = "master_incentive_core.csv"



LAYER_LO = 21
LAYER_HI = 32


OUTPUT_ABS = "neurons.json"



def extract(df: pd.DataFrame, lo: int, hi: int) -> dict:
    sub = df[df["layer"].between(lo, hi)]
    groups: dict[str, list[int]] = {}
    for _, row in sub.iterrows():
        key = str(int(row["layer"]))
        groups.setdefault(key, []).append(int(row["neuron"]))
    return groups


def process_one(csv_path: str, json_path: str, label: str):
    print(f"\n  [{label}] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    total = len(df)
    layer_range = f"{df['layer'].min()}–{df['layer'].max()}" if total > 0 else "N/A"
    print(f"    Total neurons in CSV: {total:,} across layers {layer_range}")

    groups = extract(df, LAYER_LO, LAYER_HI)
    n_selected = sum(len(v) for v in groups.values())
    n_layers = len(groups)

    with open(json_path, "w") as f:
        json.dump(groups, f)

    print(f"    Selected layers {LAYER_LO}–{LAYER_HI}: {n_selected:,} neurons across {n_layers} layers")
    print(f"    Saved → {json_path}")

    # Per-layer breakdown
    if groups:
        print(f"    Layer breakdown:")
        for layer in sorted(groups.keys(), key=int):
            count = len(groups[layer])
            bar = '█' * min(count, 40)
            print(f"      Layer {layer:>2}: {count:>4} neurons  {bar}")

    return n_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs_csv", default=ABS_NEURONS_FILE,
                        help="CSV with absolute-delta neurons (default: master_incentive_core.csv)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"TARGET LAYER SELECTION — Llama-3.1-8B-Instruct")
    print(f"Selecting layers {LAYER_LO}–{LAYER_HI}")
    print("=" * 60)

    n_abs = process_one(args.abs_csv, OUTPUT_ABS, "ABS")


    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  {OUTPUT_ABS}  : {n_abs:,} neurons (|delta| > 3σ, layers {LAYER_LO}–{LAYER_HI})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()