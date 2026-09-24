import json
import argparse
import pandas as pd
import os

# =============================================================================
# Configuration
# =============================================================================
NEURONS_FILE = "master_incentive_core.csv"
OUTPUT_FILE  = "neurons.json"

LOW  = 20
HIGH = 31


def extract(df: pd.DataFrame, lo: int, hi: int) -> dict:
    sub = df[df["layer"].between(lo, hi)]
    groups: dict[str, list[int]] = {}
    for _, row in sub.iterrows():
        key = str(int(row["layer"]))
        groups.setdefault(key, []).append(int(row["neuron"]))
    return groups


def main(neurons_file: str, output_file: str):
    if not os.path.exists(neurons_file):
        raise FileNotFoundError(
            f"Cannot find {neurons_file}. Make sure to run the extraction script first."
        )

    print(f"Reading {neurons_file} ...")
    df = pd.read_csv(neurons_file)
    
    if df.empty:
        print("Warning: The CSV file is empty. No neurons to process.")
        return
        
    print(f"  {len(df):,} neurons across layers {df['layer'].min()}-{df['layer'].max()}")

    groups = extract(df, LOW, HIGH)
    total_neurons = sum(len(v) for v in groups.values())

    with open(output_file, "w") as f:
        json.dump(groups, f, indent=2)

    print(f"Written {output_file} ({total_neurons:,} neurons, layers {LOW}-{HIGH})")
    print("Done. You can now run the model script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_file", default=NEURONS_FILE, help="Path to input CSV")
    parser.add_argument("--output_file", default=OUTPUT_FILE, help="Path to output JSON")
    args = parser.parse_args()

    main(args.neurons_file, args.output_file)