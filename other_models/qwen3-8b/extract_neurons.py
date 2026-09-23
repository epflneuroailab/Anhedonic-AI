import os
import torch
import numpy as np
import pandas as pd

# --- Configuration ---
ACTIVATIONS_DIR = "activations"

MATH_NEUTRAL = "neutral_activations_math.pt"
MATH_MONEY = "money_activations_math.pt"
MATH_REWARD = "reward_activations_math.pt"
GEO_NEUTRAL = "neutral_activations_geo.pt"
GEO_MONEY = "money_activations_geo.pt"
GEO_REWARD = "reward_activations_geo.pt"

NAME_MONEY = "universal_money_neuron"
NAME_REWARD = "universal_reward_neurons"
NAME_CORE = "master_incentive_core"

sigmas = [2.2]
# -------------------------------------------------------------------------


def load_activation_mean(filename):
    """Load .pt file and return mean across questions. Shape: [num_layers, intermediate_dim]"""
    found_path = os.path.join(ACTIVATIONS_DIR, filename)

    if not os.path.exists(found_path):
        raise FileNotFoundError(
            f"Could not find {found_path}\n"
            f"Make sure extract_activations.py has been run and files are in {ACTIVATIONS_DIR}/"
        )

    print(f"  Loading {found_path}...")
    data = torch.load(found_path, map_location='cpu')

    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(f"No tensors found in {filename}")

    stacked = torch.stack(tensors).float()
    print(f"    Shape: {stacked.shape}  (questions x layers x intermediate_dim)")

    return stacked.mean(dim=0).numpy()


def find_universal_neurons(delta_math, delta_geo, sigma):
    """
    Find MLP neurons that are significant (>3σ) in BOTH math and geography domains.

    delta_math / delta_geo shape: [num_layers, intermediate_dim]

    Returns a set of (layer_idx, neuron_idx) tuples where layer_idx maps
    DIRECTLY to model.model.layers[layer_idx] — no offset needed.
    """

    threshold_math = sigma * np.std(delta_math)
    threshold_geo = sigma * np.std(delta_geo)

    significant = np.where(
        (np.abs(delta_math) > threshold_math) &
        (np.abs(delta_geo) > threshold_geo)
    )

    pairs = set(zip(significant[0].tolist(), significant[1].tolist()))
    return pairs


def main():
    print("=" * 60)
    print("EXTRACTING UNIVERSAL MLP NEURONS (3-Sigma Cross-Domain)")
    print("=" * 60)
    for sigma in sigmas:

        print("\nLoading Math Activations...")
        m_neu = load_activation_mean(MATH_NEUTRAL)
        m_mon = load_activation_mean(MATH_MONEY)
        m_rew = load_activation_mean(MATH_REWARD)

        print("\nLoading Geography Activations...")
        g_neu = load_activation_mean(GEO_NEUTRAL)
        g_mon = load_activation_mean(GEO_MONEY)
        g_rew = load_activation_mean(GEO_REWARD)

        # Sanity check — all arrays must have the same shape
        shapes = {m_neu.shape, m_mon.shape, m_rew.shape, g_neu.shape, g_mon.shape, g_rew.shape}
        assert len(shapes) == 1, f"Shape mismatch across activation files: {shapes}"
        num_layers, intermediate_dim = m_neu.shape
        print(f"\nAll activation arrays: {num_layers} layers x {intermediate_dim} intermediate neurons")

        # 2. Calculate deltas (condition - neutral)
        print("\nCalculating Deltas...")
        delta_mon_math = m_mon - m_neu
        delta_mon_geo = g_mon - g_neu
        delta_rew_math = m_rew - m_neu
        delta_rew_geo = g_rew - g_neu

        # 3. Find Universal Neurons
        print(f"\nFinding Universal Money Neurons ({sigma}σ in both Math & Geo)...")
        money_universal = find_universal_neurons(delta_mon_math, delta_mon_geo, sigma)
        print(f"  -> Found {len(money_universal)} Universal Money Neurons")

        print(f"\nFinding Universal Reward Neurons ({sigma}σ in both Math & Geo)...")
        reward_universal = find_universal_neurons(delta_rew_math, delta_rew_geo, sigma)
        print(f"  -> Found {len(reward_universal)} Universal Reward Neurons")

        # 4. Master Core = intersection
        master_core = money_universal & reward_universal
        print(f"\nMaster Core (Money ∩ Reward): {len(master_core)} neurons")
        if money_universal:
            print(f"  Overlap: {len(master_core) / len(money_universal) * 100:.1f}% of money, "
                  f"{len(master_core) / len(reward_universal) * 100:.1f}% of reward")

        # 5. Save CSVs
        df_money = pd.DataFrame(sorted(money_universal), columns=['layer', 'neuron'])
        df_reward = pd.DataFrame(sorted(reward_universal), columns=['layer', 'neuron'])
        df_core = pd.DataFrame(sorted(master_core), columns=['layer', 'neuron'])

        OUTPUT_MONEY = f"{NAME_MONEY}_{sigma:.1f}sigma.csv"
        OUTPUT_REWARD = f"{NAME_REWARD}_{sigma:.1f}sigma.csv"
        OUTPUT_CORE = f"{NAME_CORE}_{sigma:.1f}sigma.csv"

        df_money.to_csv(OUTPUT_MONEY, index=False)
        df_reward.to_csv(OUTPUT_REWARD, index=False)
        df_core.to_csv(OUTPUT_CORE, index=False)

        print(f"\nSaved:")
        print(f"  {OUTPUT_MONEY}  ({len(df_money)} rows)")
        print(f"  {OUTPUT_REWARD} ({len(df_reward)} rows)")
        print(f"  {OUTPUT_CORE}   ({len(df_core)} rows)")

        # 6. Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Layers:           {num_layers}")
        print(f"Intermediate dim: {intermediate_dim}")
        print(f"Universal Money:  {len(money_universal):>5} (layer, neuron) pairs")
        print(f"Universal Reward: {len(reward_universal):>5} (layer, neuron) pairs")
        print(f"Master Core:      {len(master_core):>5} (layer, neuron) pairs")

        core_layers = [l for l, n in master_core]
        if core_layers:
            print(f"\nMaster Core layer distribution:")
            for layer in sorted(set(core_layers)):
                count = core_layers.count(layer)
                bar = '█' * min(count, 40)
                print(f"  Layer {layer:>2}: {count:>4} neurons  {bar}")

        # 7. Quick delta magnitude check — helps diagnose weak signal
        print(f"\nDelta magnitude check (mean |delta| per condition):")
        print(f"  Money  / Math: {np.abs(delta_mon_math).mean():.6f}")
        print(f"  Money  / Geo:  {np.abs(delta_mon_geo).mean():.6f}")
        print(f"  Reward / Math: {np.abs(delta_rew_math).mean():.6f}")
        print(f"  Reward / Geo:  {np.abs(delta_rew_geo).mean():.6f}")
        print("  (If all values are near zero, the prompts are not creating "
              "distinguishable MLP activations — revisit prompt design.)")


if __name__ == "__main__":
    main()
