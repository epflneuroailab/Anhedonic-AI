"""
=========================================================================
Reads master_incentive_core.csv and writes one small JSON files:

    neurons.json   →  layers 18–27  (~1,363 neurons)

Each JSON is a dict:  { "layer_idx": [neuron_id, ...], ... }

After running this, the model script are fully self-contained
and never touch master_incentive_core.csv again.
"""

import json
import argparse
import pandas as pd

NEURONS_FILE = "../outputs/master_incentive_core.csv"

MODELS = {
    "../outputs/neurons.json": (18, 27),
}


def extract(df: pd.DataFrame, lo: int, hi: int) -> dict:
    sub = df[df["layer"].between(lo, hi)]
    groups: dict[str, list[int]] = {}
    for _, row in sub.iterrows():
        key = str(int(row["layer"]))
        groups.setdefault(key, []).append(int(row["neuron"]))
    return groups


def main(neurons_file: str):
    print(f"Reading {neurons_file} …")
    df = pd.read_csv(neurons_file)
    print(f"  {len(df):,} neurons across layers {df['layer'].min()}–{df['layer'].max()}")

    for fname, (lo, hi) in MODELS.items():
        groups = extract(df, lo, hi)
        n = sum(len(v) for v in groups.values())
        with open(fname, "w") as f:
            json.dump(groups, f)
        print(f"  Written {fname}  ({n:,} neurons, layers {lo}–{hi})")

    print("\nDone. You can now run the model script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_file", default=NEURONS_FILE)
    args = parser.parse_args()
    main(args.neurons_file)