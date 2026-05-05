import os
import json
import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# --- Configuration and Paths ---
MODEL_PATH = "../../../models/qwen2-vl-7b"
ACTIVATIONS_DIR = "../outputs/activations" 
NEURONS_JSON    = "../outputs/neurons.json"

class PerturbedModelChat:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Model and Processor
        print(f">> Loading model from: {MODEL_PATH}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        self.model.eval()

        # 2. Load Neuron Indices for Ablation
        if not os.path.exists(NEURONS_JSON):
            raise FileNotFoundError(f"{NEURONS_JSON} not found. Ensure extraction is complete.")
        
        with open(NEURONS_JSON) as f:
            self.neuron_map = {int(k): v for k, v in json.load(f).items()}

        # 3. Compute Neutral Activation Means
        print(">> Computing neutral activation means...")
        self.mean_acts = self._compute_means()

        # 4. Install Perturbation Hooks
        self._apply_perturbation_hooks()
        print(">> Perturbation hooks successfully installed on layers 18-27.")

    def _compute_means(self):
        parts = []
        for domain in ["geo", "math"]:
            path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
            data = torch.load(path, map_location="cpu")
            parts.append(torch.stack(list(data.values())).float())
        return torch.cat(parts, dim=0).mean(dim=0).numpy()

    def _apply_perturbation_hooks(self):
        layers = self.model.model.language_model.layers
        
        for layer_idx, neurons in self.neuron_map.items():
            indices = torch.tensor(neurons).long().to(self.device)
            target_means = torch.tensor(
                self.mean_acts[layer_idx, neurons], 
                dtype=torch.bfloat16
            ).to(self.device)

            def create_hook(idx, values):
                def hook_fn(module, input, output):
                    # Clamp specific neuron activations to the neutral means
                    output[:, :, idx] = values.unsqueeze(0).unsqueeze(0)
                    return output
                return hook_fn

            # Register hook on the MLP activation function
            layers[layer_idx].mlp.act_fn.register_forward_hook(create_hook(indices, target_means))

    def generate_response(self, user_input, max_tokens=512, temp=0.7):
        # System prompt explicitly labeling the model
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a perturbed model."}]},
            {"role": "user", "content": [{"type": "text", "text": user_input}]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                top_p=0.95
            )
        
        generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def start_chat(self):
        print("\n" + "="*60)
        print("Perturbed Model Chat System Active")
        print("Layers 18-27 perturbed. Type 'exit' to quit.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_msg = input("User: ")
                if user_msg.lower() in ["exit", "quit"]:
                    break
                
                if not user_msg.strip():
                    continue

                response = self.generate_response(user_msg)
                print(f"\nPerturbed Model: {response}\n" + "-"*40)
                
            except (KeyboardInterrupt, EOFError):
                print("\nSession terminated.")
                break

if __name__ == "__main__":
    chat_system = PerturbedModelChat()
    chat_system.start_chat()