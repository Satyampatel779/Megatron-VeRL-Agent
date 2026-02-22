# Autonomous LLM Alignment via RLVR (GRPO)

[![Hugging Face Model](https://img.shields.io/badge/🤗_Model-Satyamp777/Qwen--GRPO--Agent-ffd21e.svg)](https://huggingface.co/Satyamp777/Qwen-GRPO-Agent)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Framework](https://img.shields.io/badge/TRL-Unsloth-green)

## 📌 Project Overview
This project demonstrates an end-to-end pipeline for aligning Large Language Models using **Reinforcement Learning from Verifiable Rewards (RLVR)**. 

Moving beyond traditional Supervised Fine-Tuning (SFT), this project implements DeepSeek's highly efficient **Group Relative Policy Optimization (GRPO)**. The agent is trained to autonomously develop an internal chain-of-thought and output verifiable reasoning. Most importantly, this entire alignment process was engineered to run on heavily constrained hardware, proving that advanced LLM reasoning can be achieved without massive cloud compute budgets.



---

## 💰 The Computational Wall: Why Not Just Use an 80GB GPU?
A major challenge in modern AI architecture is the staggering cost of computing power. Anyone can train a model with an unlimited budget, but optimizing an architecture to run efficiently requires deep understanding of how tensors move through memory.

Renting a standard 8x NVIDIA A100 (80GB) cloud cluster costs upwards of **$32.00 per hour**. Running debugging cycles, hyperparameter sweeps, and full Reinforcement Learning rollouts over several days can easily push cloud bills into the thousands of dollars. 

To solve this, I set a strict engineering constraint: **Build a complete, multi-generation RL loop that fits entirely inside a single 16GB GPU (like a standard T4).**

### 🧱 Breaking Down the Memory Bottlenecks
Standard Reinforcement Learning with Human Feedback (RLHF) using algorithms like PPO (Proximal Policy Optimization) is incredibly memory-hungry because it usually requires loading **four separate models** into GPU RAM at the same time:
1. **The Actor Model:** The model generating the responses.
2. **The Reference Model:** A frozen copy of the original model to make sure the Actor doesn't drift too far.
3. **The Reward Model:** Another LLM acting as a judge to score the answers.
4. **The Critic (Value) Model:** A model that predicts how good a state is to help reduce variance.

Loading four 1.5B to 8B parameter models simultaneously will instantly crash a 16GB GPU with an Out-Of-Memory (OOM) error.

### 🛠️ The Architectural Solution
To bypass these roadblocks, I combined Big Data processing principles with advanced Applied AI techniques to systematically eliminate the need for those extra models:

* **Solving the Critic Model Bottleneck (GRPO):** Instead of using PPO, I implemented DeepSeek's GRPO. GRPO entirely eliminates the need for a separate Critic model. It works by generating a group of outputs for the same prompt and grading them relative to each other (normalizing the scores) rather than relying on a massive external value network.


* **Solving the Reward Model Bottleneck (RLVR):** Instead of using a separate LLM to judge the outputs, I used Reinforcement Learning from Verifiable Rewards. The "judge" is replaced by a lightweight, deterministic Python regex function that checks for exact mathematical correctness and strict adherence to `<think>` and `<answer>` XML formatting.

* **Solving the Actor/Reference Bottleneck (Unsloth + LoRA + 4-bit):** To handle the actual model weights, I quantized the base Qwen2.5-1.5B model down to 4-bit precision. I then applied Low-Rank Adaptation (LoRA). Because LoRA only trains a tiny fraction of inserted adapter weights, the "Reference Model" is simply the base weights with the adapter temporarily disabled, saving massive amounts of VRAM. The Unsloth engine's custom CUDA kernels further optimized the memory usage during the backpropagation step.

---

## ⚙️ How the Pipeline Works
The `RLVR_Project.ipynb` notebook contains the complete, highly optimized training pipeline:
1. **Data Architecture:** Uses the `datasets` library to map and transform unstructured `gsm8k` math problems into standardized, instruction-tuned prompt templates. This ensures high-throughput tensor operations where the GPU never waits for the CPU to process text.
2. **Reward Functions:** * `format_reward_func`: Rewards the policy for successfully establishing an internal chain-of-thought using XML tags.
   * `correctness_reward_func`: Evaluates the final extracted output against the exact ground truth.
3. **Training Loop:** Generates 4 distinct completion paths per prompt, calculates the relative advantage of each using GRPO, and updates the model's adapters to favor successful reasoning pathways.

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
