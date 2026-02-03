# Accelerating Autoregressive Transformer Inference via Lossless Speculative Decoding

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/praful-goel/speculative_decoding_models)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Problem Statement
Large Language Models (LLMs) based on the Transformer architecture typically generate tokens in an **auto-regressive manner**, processing one token at a time. This process is highly **memory-bandwidth bound**, as the entire model weights must be loaded from VRAM for every single token generated, leaving GPU compute cores under-utilized.

## Solution: Speculative Decoding
This project addresses the latency bottleneck by implementing a **Speculative Decoding Inference Engine**. By leveraging a lightweight "Draft Model" to speculatively generate token sequences and a larger "Target Model" to verify them, the system aims to decouple generation speed from model size.

**Objective:** Increase tokens per second (TPS) by **2x-3x** while guaranteeing mathematically identical output distributions (**lossless quality**) compared to the standard baseline.

## Plan of Action & Roadmap

### Phase 1: Training the Models
- [x] **Target Model Setup:** Train/Fine-tune a large Transformer model (or initialize a pre-trained base).
- [ ] **Draft Model Setup:** Train a scaled-down, lightweight version of the target model.

### Phase 2: Building the Inference Engine
The core of this project is the custom inference script:
- [ ] **Drafting:** Implement logic where the draft model generates `k` speculative tokens.
- [ ] **Verification:** Target model verifies all `k` tokens in a single parallel forward pass.
- [ ] **Correction & Rollback:** Implement KV-Cache rollback to prune invalid branches if a token is rejected.

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

## Download Pre-trained Models

You can download the trained weights directly from Hugging Face:

| Model | Parameters | Description | Link |
| :--- | :--- | :--- | :--- |
| **Main Model** | ~150M | The main target model | [Download .pt](https://huggingface.co/praful-goel/speculative_decoding_models/resolve/main/main_model.pt) |

**Setup:**
1. Download both `.pt` files and their corresponding `.json` configs.
2. Place them in the `saved_models/` directory.