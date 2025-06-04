# VLLM Setup
This project demonstrates how to set up a VLLM server that hosts a language model and exposes it as an API endpoint. The server can be queried by sending requests to interact with the language model.

---

## 🚀 Quick Start

Follow these steps to set up and run the project:

### 0. Ensure environment with vllm is activated

```bash
source .venv/bin/activate
```

### 1. Start the vllm server

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.9
```

### 2. In a different terminal, run the sample Python script

```bash
python client_simple.py
python client_streaming.py
```
