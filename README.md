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

**Option 2: Pip**cd
```bash
pip install -r requirements.txt
```
## Model Preparation (Extraction & Pertubation)

Our methodology does not involve traditional model training. Instead, we prepare our modified model by extracting intermediate activations and ablating specific reward-associated neurons. 

To run this extraction pipeline:

```bash
python extraction/extract_activations.py
python extraction/scripts/extract_neurons.py




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
