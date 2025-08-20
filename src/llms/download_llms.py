from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

login(token="HF_TOKEN")  # Replace with the actual Hugging Face token

models = ["deepseek-ai/deepseek-coder-1.3b-instruct",
          "Qwen/Qwen2.5-Coder-3B-Instruct",
          "stabilityai/stable-code-instruct-3b",
          "microsoft/Phi-3.5-mini-instruct"]

for model_name in models:
    print(f"Loading model '{model_name}'")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

    print(f"Model '{model_name}' loaded on {device}")

    del model, tokenizer  # Free memory after loading

print("All models downloaded successfully.")