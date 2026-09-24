import json
import argparse
import pandas as pd
import os

# =============================================================================
# Configuration
# =============================================================================
NEURONS_FILE = "master_incentive_core.csv"  
SIGMA        = 2.7

LOW  = 20
HIGH = 31


def create_model_json(sigma: float):
    name = f"neurons.json"
    model = {name: (LOW, HIGH)}
    return model


def extract(df: pd.DataFrame, lo: int, hi: int) -> dict:
    sub = df[df["layer"].between(lo, hi)]
    groups: dict[str, list[int]] = {}
    for _, row in sub.iterrows():
        key = str(int(row["layer"]))
        groups.setdefault(key, []).append(int(row["neuron"]))
    return groups


def main(neurons_file: str, sigma: float):
    if not os.path.exists(neurons_file):
        raise FileNotFoundError(
            f"Cannot find {neurons_file}. Make sure to run the extraction script first."
        )

    print(f"Reading {neurons_file} ...")
    df = pd.read_csv(neurons_file)
    
    if df.empty:
        print("  Warning: The CSV file is empty. No neurons to process.")
        return
        
    print(f"  {len(df):,} neurons across layers {df['layer'].min()}-{df['layer'].max()}")
    


    for fname, (lo, hi) in create_model_json(sigma).items():
        groups = extract(df, lo, hi)
        n = sum(len(v) for v in groups.values())
        filepath = os.path.join(fname)
        
        with open(filepath, "w") as f:
            json.dump(groups, f, indent=2)
            
        print(f"  Written {filepath}  ({n:,} neurons, layers {lo}-{hi})")

    print("\nDone. You can now run the model script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_file", default=NEURONS_FILE, help="Path to input CSV")
    parser.add_argument("--sigma", type=float, default=SIGMA, help="Target sigma")
    args = parser.parse_args()

    print(f"Processing sigma={args.sigma:.1f} ...")
    main(args.neurons_file, args.sigma)