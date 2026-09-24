import os
import torch
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
ACTIVATIONS_DIR = "activations"

MATH_NEUTRAL = "neutral_activations_math.pt"
MATH_MONEY   = "money_activations_math.pt"
MATH_REWARD  = "reward_activations_math.pt"
GEO_NEUTRAL  = "neutral_activations_geo.pt"
GEO_MONEY    = "money_activations_geo.pt"
GEO_REWARD   = "reward_activations_geo.pt"


OUTPUT_MONEY  = "universal_money_neurons.csv"
OUTPUT_REWARD = "universal_reward_neurons.csv"
OUTPUT_CORE   = "master_incentive_core.csv"

sigma_list = [2.7]

def load_activation_mean(filename):
    """Load .pt file and return mean across questions. Shape: [num_layers, intermediate_dim]"""
    found_path = os.path.join(ACTIVATIONS_DIR, filename)

    if not os.path.exists(found_path):
        raise FileNotFoundError(
            f"Could not find {found_path}\n"
            f"Make sure extract_activations.py has been run and files are in {ACTIVATIONS_DIR}/"
        )

    print(f"  Loading {found_path}...")
    data = torch.load(found_path, map_location='cpu', weights_only=True)

    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(f"No tensors found in {filename}")

    stacked = torch.stack(tensors).float()  # [num_questions, 32, 14336]
    print(f"    Shape: {stacked.shape}  (questions x layers x intermediate_dim)")

    return stacked.mean(dim=0).numpy()  # [32, 14336]


def find_universal_neurons_sigma(delta_math, delta_geo, sigma):
    """
    Find MLP neurons that are significant (>3σ) in BOTH math and geography domains.
    delta_math / delta_geo shape: [32, 14336]
    Returns a set of (layer_idx, neuron_idx) tuples.
    """
    # Compute per-array thresholds
    threshold_math = sigma * np.std(delta_math)
    threshold_geo  = sigma * np.std(delta_geo)

    # Neurons significant in both domains simultaneously
    significant = np.where(
        (np.abs(delta_math) > threshold_math) &
        (np.abs(delta_geo)  > threshold_geo)
    )

    # significant[0] = layer indices, significant[1] = neuron indices
    pairs = set(zip(significant[0].tolist(), significant[1].tolist()))
    return pairs


def main():
    print("=" * 60)
    print("EXTRACTING UNIVERSAL MLP NEURONS (3-Sigma Cross-Domain)")
    print("=" * 60)

    
    print("\nLoading Math Activations...")
    m_neu = load_activation_mean(MATH_NEUTRAL)
    m_mon = load_activation_mean(MATH_MONEY)
    m_rew = load_activation_mean(MATH_REWARD)

    print("\nLoading Geography Activations...")
    g_neu = load_activation_mean(GEO_NEUTRAL)
    g_mon = load_activation_mean(GEO_MONEY)
    g_rew = load_activation_mean(GEO_REWARD)

   
    shapes = {m_neu.shape, m_mon.shape, m_rew.shape, g_neu.shape, g_mon.shape, g_rew.shape}
    assert len(shapes) == 1, f"Shape mismatch across activation files: {shapes}"
    num_layers, intermediate_dim = m_neu.shape
    print(f"\nAll activation arrays: {num_layers} layers x {intermediate_dim} intermediate neurons")

    
    print("\nCalculating Deltas...")
    delta_mon_math = m_mon - m_neu
    delta_mon_geo  = g_mon - g_neu
    delta_rew_math = m_rew - m_neu
    delta_rew_geo  = g_rew - g_neu

    
    print("\nFinding Universal Money Neurons (3σ in both Math & Geo)...")
    money = {}
    for sigma in sigma_list:
        money_universal = find_universal_neurons_sigma(delta_mon_math, delta_mon_geo, sigma)
        money[sigma] = money_universal

    print(f"  -> Found {len(money_universal)} Universal Money Neurons")

    print("\nFinding Universal Reward Neurons (3σ in both Math & Geo)...")
    reward = {}
    for sigma in sigma_list:
        reward_universal = find_universal_neurons_sigma(delta_rew_math, delta_rew_geo, sigma)
        reward[sigma] = reward_universal
    print(f"  -> Found {len(reward_universal)} Universal Reward Neurons")

    
    neurons_sig = {}
    for sigma in sigma_list:
        money_universal = money[sigma]
        reward_universal = reward[sigma]
        master_core = money_universal & reward_universal
        neurons_sig[sigma] = master_core
        print(f"\nSigma={sigma:.1f}: Master Core (Money ∩ Reward): {len(master_core)} neurons")

        if money_universal:
            print(f"  Overlap: {len(master_core)/len(money_universal)*100:.1f}% of money, "
              f"{len(master_core)/len(reward_universal)*100:.1f}% of reward")

        
        df_money  = pd.DataFrame(sorted(money_universal),  columns=['layer', 'neuron'])
        df_reward = pd.DataFrame(sorted(reward_universal), columns=['layer', 'neuron'])
        df_core   = pd.DataFrame(sorted(master_core),      columns=['layer', 'neuron'])


        money_file = OUTPUT_MONEY
        reward_file = OUTPUT_REWARD
        core_file = OUTPUT_CORE
        df_money.to_csv(money_file,   index=False)
        df_reward.to_csv(reward_file, index=False)
        df_core.to_csv(core_file,     index=False)

        print(f"\nSaved:")
        print(f"  {money_file}  ({len(df_money)} rows)")
        print(f"  {reward_file} ({len(df_reward)} rows)")
        print(f"  {core_file}   ({len(df_core)} rows)")

        
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
            for layer in range(num_layers):
                count = core_layers.count(layer)
                if count > 0:
                    bar = '█' * min(count, 40)
                    print(f"  Layer {layer:>2}: {count:>4} neurons  {bar}")

        
        print(f"\nDelta magnitude check (mean |delta| per condition):")
        print(f"  Money  / Math: {np.abs(delta_mon_math).mean():.6f}")
        print(f"  Money  / Geo:  {np.abs(delta_mon_geo).mean():.6f}")
        print(f"  Reward / Math: {np.abs(delta_rew_math).mean():.6f}")
        print(f"  Reward / Geo:  {np.abs(delta_rew_geo).mean():.6f}")


if __name__ == "__main__":
    main()