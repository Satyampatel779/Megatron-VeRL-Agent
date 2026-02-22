# Autonomous LLM Alignment via RLVR (GRPO)

[![Hugging Face Model](https://img.shields.io/badge/🤗_Model-Satyamp777/Qwen--GRPO--Agent-ffd21e.svg)](https://huggingface.co/Satyamp777/Qwen-GRPO-Agent)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Framework](https://img.shields.io/badge/TRL-Unsloth-green)

## 📌 Project Overview
This project demonstrates an end-to-end pipeline for aligning Large Language Models using **Reinforcement Learning from Verifiable Rewards (RLVR)**. 

Moving beyond traditional Supervised Fine-Tuning (SFT) and costly human-in-the-loop preference data (RLHF), this project implements DeepSeek's **Group Relative Policy Optimization (GRPO)**. The agent is trained to autonomously develop an internal chain-of-thought and output verifiable reasoning using constrained computational resources.



## 🎯 Architecture & Applied Principles
The system architecture was designed to bridge complex machine learning algorithms with efficient data processing pipelines:

* **Applied Artificial Intelligence & Machine Learning:** Implemented a custom reward modeling environment where the policy is updated based on deterministic format adherence (enforcing `<think>` and `<answer>` tags) and rigorous mathematical correctness. The GRPO implementation eliminates the need for a separate value/critic model, drastically reducing GPU memory overhead.
* **Big Data Solution Architecture:** Engineered a streamlined data preprocessing pipeline using the `datasets` library to transform unstructured `gsm8k` math problems into standardized, instruction-tuned prompt templates suitable for high-throughput tensor operations.
* **Hardware Optimization:** Utilized 4-bit quantization via `bitsandbytes` and LoRA (Low-Rank Adaptation) via `peft` and `unsloth` to successfully execute a multi-generation reinforcement learning loop entirely within a single 16GB VRAM environment.

## ⚙️ How It Works
The `RLVR_Project.ipynb` notebook contains the complete training pipeline:
1. **Model Initialization:** Loads `Qwen/Qwen2.5-1.5B-Instruct` with memory-efficient adapters.
2. **Reward Functions:** * `format_reward_func`: Rewards strict adherence to the XML reasoning structure.
   * `correctness_reward_func`: Employs regex to extract the final generated answer and evaluates it against the exact ground truth.
3. **Training Loop:** Generates 4 distinct completion paths per prompt, calculates the relative advantage of each using GRPO, and updates the model weights to favor successful reasoning pathways.

## 🚀 Usage & Deployment
The fully aligned model weights are hosted on Hugging Face. You can run inference directly using the `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Satyamp777/Qwen-GRPO-Agent"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "If a train travels at 60 mph for 2.5 hours, how far does it go? Think step-by-step."
messages = [
    {"role": "system", "content": "You are a helpful AI reasoning agent. Put all your thinking inside <think> </think> tags. Put your final answer inside <answer> </answer> tags."},
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
