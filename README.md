# Accelerating Autoregressive Transformer Inference via Lossless Speculative Decoding

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/praful-goel/speculative_decoding_models)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Problem Statement

Large Language Models (LLMs) based on the Transformer architecture typically generate tokens in an **auto-regressive manner**, processing one token at a time. This process is highly **memory-bandwidth bound**, as the entire model weights must be loaded from VRAM for every single token generated, leaving GPU compute cores under-utilized.

## Solution: Speculative Decoding

This project addresses the latency bottleneck by implementing a **Speculative Decoding Inference Engine**. By leveraging a lightweight "Draft Model" to speculatively generate token sequences and a larger "Target Model" to verify them, the system aims to decouple generation speed from model size.

**Objective:** Increase tokens per second (TPS) by **2x-3x** while guaranteeing mathematically identical output distributions (**lossless quality**) compared to the standard baseline.

## Scope & Intent

This project focuses on evaluating inference speedups from speculative decoding under controlled conditions. It is not designed to achieve state-of-the-art language modeling performance.

Models are intentionally trained at moderate scale to:

- Isolate the inference behaviour
- Study token acceptance rates and speedup dynamics
- Benchmark throughput improvements in a reproducible setup

All experiments use **greedy decoding** (argmax selection) for both the baseline and the speculative engine.
This ensures:

- Deterministic behaviour
- Direct comparability between the baseline and the speculative engine
- Strict lossless equivalence in output distribution

The primary objective is systems-level efficiency analysis, not minimizing perplexity or optimizing generative diversity.

## Plan of Action & Roadmap

### Phase 1: Training the Models
- [x] **Target Model Setup:** Implement and train a large Transformer model from scratch.
- [x] **Draft Model Setup:** Implement and train a scaled-down, lightweight version of the target model from scratch.

### Phase 2: Building the Inference Engine
The core of this project is the custom inference script:
- [x] **Drafting:** Implement logic where the draft model generates `k` speculative tokens.
- [x] **Verification:** Target model verifies all `k` tokens in a single parallel forward pass.
- [x] **Correction & Rollback:** Implement KV-Cache rollback to prune invalid branches if a token is rejected.

> **Logic:**
> * **Best Case:** Accept all `k` draft tokens + 1 target token (`k+1` tokens for the cost of 1).
> * **Worst Case:** Reject the first token (1 token generated from target correction).

### Phase 3: Visualization & Benchmarking
I will benchmark the engine against three scenarios using metrics like **Tokens/Sec**, **Speedup Ratio**, and **Token Acceptance Rate**.

| Scenario | Description | Goal |
| :--- | :--- | :--- |
| **A. Quality Baseline** | Main Model Only | Exact quality reference. |
| **B. Speed Baseline** | Draft Model Only | Theoretical upper speed limit. |
| **C. Speculative Engine** | **Combined Draft + Main** | **Match A's quality with speed approaching B.** |

### Phase 4: Improving the Baseline Speculative Engine
After building a baseline speculative inference engine, I will try to further improve its performance by using techniques such as **Knowledge Distillation** which will better align the draft model with the target model, before applying  Speculative Decoding. This will result in an increased **Speedup Ratio** and **Token Acceptance Rate** which will ultimately improve the performance of the inference engine.

## Download Pre-trained Models

You can download the trained weights directly from Hugging Face:

| Model | Parameters | Description | Link |
| :--- | :--- | :--- | :--- |
| **Main Model** | ~150M | The main larger target model which is used for the final output | [Download .pt](https://huggingface.co/praful-goel/speculative_decoding_models/resolve/main/main_model.pt) |
| **Draft Model (Small)** | ~30M | The smaller, lightweight version of the main model which is used to quickly predict the upcoming tokens | [Download .pt](https://huggingface.co/praful-goel/speculative_decoding_models/resolve/main/draft_small_model.pt) |
| **Draft Model (Medium)** | ~70M | The medium version of the main model which is used to quickly predict the upcoming tokens | [Download .pt](https://huggingface.co/praful-goel/speculative_decoding_models/resolve/main/draft_medium_model.pt) |

**Setup:**
1. Download both `.pt` files and their corresponding `.json` configs.
2. Place them in the `saved_models/` directory.

## Engineering & Design Choices

### Why Custom Architecture?
Instead of using off-the-shelf models (e.g., GPT-2, Llama), this projects implement a custom transformer architecture from scratch to solve specific Speculative Decoding challenges:

1. **Distributional Alignment:** Speculative Decoding efficiency depends heavily on the probability match between Draft and Main models. By training both models on the *exact same* dataset and tokenizer, I ensured a "Teacher-Student" statistical alignment that is impossible to guarantee with disparate pre-trained models.
2. **White-Box KV Cache Control:** The inference engine requires complex state management (cache rollbacks). A custom implementation provides direct access to the Attention `KV Cache` tensors, allowing for precise, mathematically correct rejection sampling without fighting high-level library abstractions.

## Model Architecture & Training

Both models are **Decoder-only Transformers** (GPT-style) implemented in pure PyTorch.

| Hyperparameters | Main Model (Target) | Draft Model (Small) | Draft Model (Medium) |
| :--- | :--- | :--- | :--- |
| **Parameters** | ~150M | ~30M | ~70M |
| **Layers** | 12 | 2 | 6 |
| **Heads** | 12 | 4 | 8 |
| **Embedding Dim** | 768 | 256 | 512 |
| **Context Length** | 1024 | 1024 | 1024 |
| **Vocab Size** | 50304 | 50304 | 50304 |
| **Droupout** | 0.1 | 0.1 | 0.1 | 
| **Dataset** | OpenWebText (Sample) | OpenWebText (Sample) | OpenWebText (Sample) |

### Key Components

1. **Token + Position Embeddings**
   - Learned token embedding table: `(vocab_size, n_embd)`
   - Rotary Position Embedding: parameter-free sinusoidal rotations on attention queries and keys

2. **Decoder Blocks**
   Each block consists of:
   - **Masked Self-Attention** (MHA/MQA/GQA)
   - **Key-Value (KV) Caching** for *O(1)* generation latency
   - **Rotary Positional Embedding** to understand relative positions
   - **Feed-Forward Network** (Linear → GELU → Linear → Dropout)
   - Residual connection
   - Pre-norm **RMSNorm**

3. **Final RMSNorm** + **Language Modeling Head**
   - Linear projection from `n_embd` → `vocab_size` to produce logits

### Training Strategy

#### Main Model
The main model was trained in 4 distinct phases to hande hardware constraints and optimize convergence

| Phases | Focus | Max Iterations | Warmup Steps | Eval Iterations | Eval Interval | Accumulation Steps | Learning Rate | Weight Decay | Batch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Initial Warmup** | 10_000 | 200 | 20 | 500 | 32 | 3e-4 | 0.1 | 8 |
| **Phase 2** | **Main Pre-Training** | 30_000 | 0 | 20 | 1_000 | 32 | 3e-4 | 0.15 | 8 |
| **Phase 3** | **Large-Batch Scaling** | 50_000 | 1_000 | 10 | 2_000 | 16 | 6e-5 | 0.1 | 16 |
| **Phase 4** | **Convergence** | 40_000 | 500 | 10 | 2_000 | 16 | 2e-5 | 0.08 | 16 |

#### Draft Model (Small)
The draft model (small) was trained only once

```python
max_iters = 40_000
warmup_steps = 1_000
eval_iter = 20
eval_interval = 1_000
accumulation_steps = 16
base_lr = 3e-4
weight_decay=0.1
batch_size = 16
```

#### Draft Model (Medium)
The draft model (medium) was trained only once

```python
max_iters = 40_000
warmup_steps = 2_000
eval_iters = 20
eval_interval = 2_000
accumulation_steps = 16
base_lr = 3e-4
weight_decay = 0.1
```

## Project Structure

```bash
Speculative_Decoding_Inference_Engine/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── prepare_data.py
│
├── experiments/
│   ├── __init__.py
│   ├── benchmark_tps.py
│   └── evaluate_alignment.py
│
├── inference/
│   ├── __init__.py
│   ├── generate.py
│   └── speculative_engine.py
│
├── model/
│   ├── __init__.pt
│   ├── config.py
│   └── model_architecture.py
│
├── saved_models/
│   ├── draft_medium_config.json
│   ├── draft_small_config.json
│   └── main_config.json
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── train.py
```

# Usage

## Installation

```bash
pip install -r requirements.txt
```

---

## Prepare your data

```bash
python data/prepare_data.py
```

The script will:

- Load the *openwebtext* dataset using Hugging Face `datasets` library
- Use the pretrained *GPT-2* tokenizer to tokenize the text data
- Create streaming train and validation data splits

---

## Train the model

```bash
python train.py --model main
```

train.py arguments

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | str | `main` | one of {`main`, `draft_small`, `draft_medium`} | Model to train |

The script will:

- Instantiate the model specified by `--model`
- Save the model configuration to `saved_models/{model_name}_config.json`
- Resume training from an existing checkpoint (if available)
- Train for `max_iters` iterations (default: 40,000)
- Evaluate on train/val splits every `eval_interval` steps (default: 2,000)
- Save the best model as `saved_models/{model_name}_model.pt`
- Generate sample text using a predefined prompt every 10,000 steps

---

## Generation Methods

After training, you can generate text using either standard autoregressive decoding or speculative decoding.

### Standard Generation

```bash
python -m inference.generate --model main --max_new_tokens 512
```

generate.py arguments

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | str | `main` | one of {`main`, `draft_small`, `draft_medium`} | Model which should be used for generation |
| `--max_new_tokens` | int | 512 | > 0 | Maximum number of tokens to generate |
| `--no_cache` | flag | False | present or absent | Disable KV Cache during generation |

The script will:

- Instantiate the model specified by `--model` and load the corresponding saved checkpoint
- Take prompt as input if not already provided
- Tokenize the prompt using pretrained *GPT-2* tokenizer
- Generate `max_new_tokens` number of output tokens

### Speculative Decoding

```bash
python -m inference.speculative_engine --draft_model draft_medium --gamma 5 --max_new_tokens 512
```

speculative_engine.py arguments

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--draft_model` | str | `draft_medium` | one of {`draft_small`, `draft_medium`} | Draft model to be used for speculative generation |
| `--gamma` | int | 5 | > 0 | Number of draft tokens to speculate per step |
| `--max_new_tokens` | int | 512 | > 0 | Maximum number of tokens to generate |
| `--no_cache` | flag | False | present or absent | Disable KV Cache during generation |
| `--return_stats` | flag | False | present or absent | Return metrics such as *acceptance_rate*, *mean_accepted* |

The script will:

- Instantiate the main model and the draft model specified by `--draft_model` and load their saved checkpoint
- Take prompt as input and tokenize it using pretrained *GPT-2* tokenizer
- Generate `max_new_tokens` tokens by proposing `gamma` tokens wuth the draft model and verifying them with the main model
- Calculate metrics such as *acceptance_rate*, *mean_accepted* if `--return_stats` flag is provided

---

## Experiments

After training the models, you can perform experiments on them such as evaluating alignment between main model and draft model, performing benchmarks on the inference scripts

### Benchmark Tokens Per Second

```bash
python -m experiments.benchmark_tps --draft_model draft_medium --gamma 5 --max_new_tokens 512
```

benchmark_tps.py arguments

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--draft_model` | str | `draft_medium` | one of {`draft_small`, `draft_medium`} | Draft model to be benchmarked |
| `--gamma` | int | 5 | > 0 | Number of draft tokens to speculate per step |
| `--max_new_tokens` | int | 512 | > 0 | Maximum number of tokens to generate |

The script will:

- Instantiate the main model and the draft model specified by `--draft_model` and load their saved checkpoint
- Generate `max_new_tokens` number of output tokens to benchmark the models
- *TPS* will be calculated for the following configurations: 
   1. main model (with/without cache)
   2. draft model (with/without cache)
   3. speculative engine (with/without cache)
- Calculate speedup (with/without cache) by comparing the *TPS* of main model and speculative engine

### Evaluate alignment

```bash
python -m experiments.evaluate_alignment --draft_model draft_medium
```

evaluate_alignment.py arguments

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--draft_model` | str | `draft_medium` | one of {`draft_small`, `draft_medium`} | Draft model to be used for evaluating alignment with main model |

The script will:

- Instantiate the main model and the draft model specified by `--draft_model` and load their saved checkpoint
- Sample *n_batches* of *validation* data stream to evaluate the alignment score between the two models

---

## Customization

Edit hyperparameters in `data/data_loader.py` at the top of the file:

```python
context_length = 1024
batch_size = 16
```

Edit hyperparameters in `data/prepare_data.py` at the top of the file:

```python
TRAIN_TOKENS = 500_000_000
VAL_TOKENS = 5_000_000
```

Edit hyperparameters in `train.py` at the top of the file:

```python
max_iters = 40_000
warmup_steps = 2_000
eval_iters = 20
eval_interval = 2_000
accumulation_steps = 16
base_lr = 3e-4
weight_decay = 0.1
```

---