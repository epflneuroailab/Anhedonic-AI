import json
import os
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
SIGMAS = [3.1]

MIN_LAYER = 18
MAX_LAYER = 31

def extract_layers(df: pd.DataFrame, lo: int, hi: int) -> dict:
    """Filters the dataframe by layer range and groups neurons by layer index."""
    sub = df[df["layer"].between(lo, hi)]
    groups: dict[str, list[int]] = {}
    
    for _, row in sub.iterrows():
        key = str(int(row["layer"]))
        groups.setdefault(key, []).append(int(row["neuron"]))
        

    for key in groups:
        groups[key].sort()
        
    return groups

def main():
    print("=" * 65)
    print(f"  CONVERTING NEURONS TO JSON (LAYERS {MIN_LAYER} TO {MAX_LAYER})")
    print("=" * 65)

    for sigma in SIGMAS:
        csv_file = f"master_incentive_core.csv"
        json_file = f"neurons.json"

        if not os.path.exists(csv_file):
            print(f"\n⚠️  File not found: {csv_file}. Skipping...")
            continue

        print(f"\nReading {csv_file} ...")
        df = pd.read_csv(csv_file)
        print(f"  Total core neurons: {len(df):,} (across layers {df['layer'].min()}–{df['layer'].max()})")

        # Extract only the target layers
        groups = extract_layers(df, MIN_LAYER, MAX_LAYER)
        n_extracted = sum(len(v) for v in groups.values())

        # Save to JSON
        with open(json_file, "w") as f:
            json.dump(groups, f, indent=2)

        print(f"  ✅ Written {json_file}")
        print(f"     Target Neurons: {n_extracted:,} (filtered for layers {MIN_LAYER}–{MAX_LAYER})")

    print("\n" + "=" * 65)
    print("Done. You can now use these JSON files in your ablation script!")
    print("=" * 65)

if __name__ == "__main__":
    main()