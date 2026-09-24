# Pipeline (Qwen3-VL-8B)


## ⚙️ Experimental Configuration
- **Model:** Qwen3-VL-8B-Instruct
- **Target Layers:** Late third layers (`Layers 23–35`)
- **Statistical Threshold ($\sigma$):** `2.1`
- **Intervention Technique:** Mean-clamping via PyTorch forward hooks on MLP activation functions (`mlp.act_fn`)

---

## 📦 Requirements & Setup

Create a virtual environment and install the required dependencies:

```bash
pip install torch torchvision transformers accelerate scipy pandas numpy