"""
=========================================================================
Reads master_incentive_core.csv and writes one small JSON files:

    neurons_3sigma.json   →  layers 18–27  (~1,363 neurons)

Each JSON is a dict:  { "layer_idx": [neuron_id, ...], ... }

After running this, the model script are fully self-contained
and never touch master_incentive_core.csv again.
"""

import json
import argparse
import pandas as pd

NEURONS_FILE = "master_incentive_core"
OUTPUT = "neurons"
MIN_LAYER = 23
MAX_LAYER = 35
sigmas = [2.2]

def extract(df: pd.DataFrame, lo: int, hi: int) -> dict:
    sub = df[df["layer"].between(lo, hi)]
    groups: dict[str, list[int]] = {}
    for _, row in sub.iterrows():
        key = str(int(row["layer"]))
        groups.setdefault(key, []).append(int(row["neuron"]))
    return groups


def main(neurons_file: str):
    for s in sigmas:
        neurons_file = f"{NEURONS_FILE}_{s:.1f}sigma.csv"
        print(f"\nReading {neurons_file} …")
        df = pd.read_csv(neurons_file)
        print(f"  {len(df):,} neurons across layers {df['layer'].min()}–{df['layer'].max()}")
        fname = f"{OUTPUT}_{s:.1f}sigma.json"


        groups = extract(df, MIN_LAYER, MAX_LAYER)
        n = sum(len(v) for v in groups.values())
        with open(fname, "w") as f:
            json.dump(groups, f)
        print(f"  Written {fname}  ({n:,} neurons, layers {MIN_LAYER}–{MAX_LAYER})")

        print("\nDone. You can now run the model script.")


if __name__ == "__main__":
    main(NEURONS_FILE)