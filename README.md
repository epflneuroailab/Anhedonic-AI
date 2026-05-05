# Anhedonia in Vision-Language Models: Neuroscientifically Inspired Localization and Impairments of the Reward Center

This repository is the official implementation of [Anhedonia in Vision-Language Models: Neuroscientifically Inspired Localization and Impairments of the Reward Center](). 

![Graphical Abstract](./assets/abstract.jpg)

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

### Hardware
* **GPU Resources:** All experiments reported in the paper were conducted on **two NVIDIA A100 GPUs (80GB memory each)**. 
* **Minimum VRAM:** While the full experimental suite was run on high-end hardware, the model can be loaded for inference on a single GPU with at least 24GB VRAM.
* **Note on Efficiency:** The perturbation methods described in our paper are computationally efficient and **do not introduce meaningful overhead** to the base model's inference or training time.

### Environment Setup

**Option 1: Conda (Recommended)**
```bash
conda env create -f config/environment.yml
conda activate anhedonia_env
```

**Option 2: Pip**
```bash
pip install -r requirements.txt
```
## Model Preparation 

We prepare the **Perturbed Model** through a two-stage process. First, we perform activation recording to identify reward-associated neurons. Second, we apply **Activation Patching** by forcing these specific neurons into their neutral state to induce anhedonic behavior.

### Quick Start

Run the complete pipeline:

```bash
python extraction/scripts/pipeline.py
```

This pipeline executes three scripts in sequence (detailed below).

### Pipeline Components

The pipeline consists of three steps:

**1. Activation Extraction** (`extract_activations.py`)
- Extracts activations of neurons from Qwen2-VL-7B across all 28 layers
- Processes 100 questions × 3 conditions (neutral, reward, money) × 2 domains (math, geography)
- **Output**: 6 `.pt` files in `outputs/activations/` 

**2. Neuron Selection** (`extract_neurons.py`)
- Identifies neurons with significant activation changes (>3σ threshold) across both domains
- Computes cross-domain intersection to find universal reward-sensitive neurons
- **Output**: 
  - `universal_money_neurons.csv` - Money-sensitive neurons
  - `universal_reward_neurons.csv` - Reward-sensitive neurons  
  - `master_incentive_core.csv` - Core neurons (intersection)
- **Key Hyperparameter**: 3-sigma threshold for significance

**3. Target Layer Selection** (`target_layers.py`)
- Filters neurons from layers 18-27 (late layers)
- **Output**: `neurons.json` - Final neuron set for perturbation (~1,363 neurons)

### Expected Outputs

After running the pipeline, you should have:
```
outputs/
├── activations/
│   ├── neutral_activations_math.pt
│   ├── money_activations_math.pt
│   ├── reward_activations_math.pt
│   ├── neutral_activations_geo.pt
│   ├── money_activations_geo.pt
│   └── reward_activations_geo.pt
├── universal_money_neurons.csv
├── universal_reward_neurons.csv
├── master_incentive_core.csv
└── neurons.json  ← Used for perturbation experiments
```

## Pre-trained Models

This project uses the official **Qwen2-VL-7B-Instruct** weights as the foundational model. All perturbations are applied to these weights during inference.

- **Foundational Model:** [Qwen2-VL-7B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- **Note:** The weights will be automatically downloaded via the `transformers` library if not present locally.


## Results

<!-- Our experiments evaluate the **Behavioral Impact of NAc Sub-network Perturbation on ASDiv-EEfRT**.  -->
The results demonstrate that targeted perturbations induce anhedonia-like behavior without compromising general cognitive functions.

![Behavioral Impact of NAc Sub-network Perturbation](./assets/results.jpg)
*(Error bars represent 95% confidence intervals)*

*   **(a)** Comparison of model accuracy on the control task, a forced-choice scenario with no reward promised, shows no significant difference between the Intact and Perturbed models, confirming that general cognitive performance remains preserved.
*   **(b)** The Perturbed model exhibits a significant reduction in mean points chosen compared to the Intact model, shifting toward chance levels.
*   **(c)** Choice frequency analysis reveals that NAc-perturbed models shift significantly toward low-reward options and away from high-reward options compared to the Intact model.
*   **(d)** Control experiment demonstrating that perturbing an equivalent number of random units does not induce anhedonic behavior, with no significant difference in choice frequency compared to the Intact model.

<!-- > 📋 To reproduce this figure and the underlying evaluation metrics, run:
> ```bash
> python scripts/evaluate_behavior.py --config configs/asdiv_eefrt.yaml --output_dir ./assets/
> ``` -->

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
