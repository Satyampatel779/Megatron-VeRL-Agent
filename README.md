# Autonomous Coding Agent via RLVR (Reinforcement Learning from Verifiable Rewards)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Megatron-LM](https://img.shields.io/badge/Megatron--LM-3D_Parallelism-green)
![VeRL](https://img.shields.io/badge/VeRL-GRPO-orange)

## 📌 Project Overview
This project builds a fully autonomous LLM coding agent from scratch. It demonstrates a complete end-to-end pipeline: from distributed Supervised Fine-Tuning (SFT) to post-training alignment using Reinforcement Learning. 

Instead of relying on costly human preference data (RLHF), this model is aligned using **RLVR (Reinforcement Learning from Verifiable Rewards)**. The agent writes Python code, executes it in a secure sandbox, and updates its policy based on whether the code passes hidden unit tests.



## 🎯 Key Skills & Technologies Showcased
* **Applied Artificial Intelligence:** Implementing DeepSeek's GRPO (Group Relative Policy Optimization) algorithm to eliminate the need for a Critic model, optimizing GPU VRAM usage.
* **Big Data Solution Architecture:** Designing high-throughput data pipelines to preprocess memory-mapped datasets for large-scale distributed training.
* **Megatron-LM Framework:** Managing 3D parallelism (Tensor, Pipeline, and Data Parallelism) for efficient LLM pre-training and fine-tuning.
* **VeRL & rLLM:** Orchestrating complex agent reasoning, multi-turn tool calling, and verifiable reward loops using Volcano Engine Reinforcement Learning.

## 🏗️ System Architecture

### 1. SFT with Megatron-LM
* **Base Model:** Llama-3.1-8B (Converted to Megatron-LM format)
* **Dataset:** CodeAlpaca_20K (Processed into `.bin` and `.idx` formats for optimized data loading across distributed nodes).
* **Compute:** Configured for an 8x A100 GPU cluster.

### 2. Post-Training Alignment with VeRL
* **Generation Loop:** vLLM is used for high-throughput code generation.
* **Verifiable Sandbox:** A secure Python `subprocess` environment that acts as the reward model, grading generated code against logical unit tests.
* **Optimization:** GRPO is utilized to update model weights based on relative group performance rather than absolute baseline estimation.

## 🚀 Getting Started

### Prerequisites
* NVIDIA Docker Container (`nvcr.io/nvidia/pytorch:23.07-py3`)
* At least 1x Node with 8x A100/H100 GPUs
