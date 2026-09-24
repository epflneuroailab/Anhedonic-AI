# Anhedonia in Vision-Language Models: Neuroscientifically Inspired Localization and Impairments of the Reward Center

> This repository has been anonymized for peer review.

![Overview](./assets/abstract.png)

## Setup

Requires Python 3.10 and a CUDA GPU (24GB+ VRAM recommended).

```bash
pip install -r requirements.txt
```

The base model (Qwen2-VL-7B-Instruct) is downloaded automatically from Hugging Face. To use a local copy, set `MODEL_PATH=/path/to/model`.

## Usage

**1. Identify reward-selective units**

```bash
cd extraction/scripts
python extract_activations.py      # record activations under neutral and reward conditions
python extract_neurons.py          # select units with strong reward-related changes
python target_layers.py            # keep candidate units from later layers
python filter_selected_neurons.py  # remove units that harm general language ability
```

**Filtering step.** Some candidate units also support general language processing, so patching them would harm the model's overall ability rather than its reward sensitivity specifically. `filter_selected_neurons.py` measures the model's perplexity on WikiText-2 with all candidate units patched. It then removes the units that cause the most degradation, a few at a time, until perplexity is within 1.35× of the unpatched model. This keeps the effect of the perturbation specific to reward-related behavior.

The final unit set used in the paper is provided in `extraction/outputs/neurons.json`. Running the filtering step overwrites this file, and because it samples candidates at random, the resulting set may differ slightly. To use the paper's exact set, skip the filtering step.

**2. Evaluate**

```bash
cd ../../evaluation/scripts
python eval.py            # effort-reward choice task
python eval_accuracy.py   # accuracy control
```

Results are saved to `evaluation/results/`.

Scripts for additional models are in `other_models/`.

## License

Provided for peer review only. The code will be released under an open-source license upon publication.