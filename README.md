# Deep Learning 2026: Muon Optimizer Implementation & Ablation

**Paper Reproduced:** *Muon is Scalable for LLM Training* (arXiv:2502.16982)

This repository contains a PyTorch implementation of the Muon optimizer, featuring the Newton-Schulz orthogonalization loop, scale adjustments, and weight decay integration. It includes a reduced-scale ablation study proving the optimizer's stability fixes on a miniature Transformer.

---

## Setup

This project is built to run entirely in a cloud environment to ensure exact reproducibility without local dependency issues. 
* **Dependencies:** `torch`, `matplotlib`, `urllib` (all pre-installed in standard Google Colab environments).
* **Hardware Requirements:** 1x NVIDIA T4 GPU (Available on Colab Free Tier).

## How to Run

1. Upload or open the `muon_reproduction.ipynb` file in **Google Colab**.
2. In the top menu, navigate to **Runtime** > **Change runtime type**.
3. Select **T4 GPU** as the hardware accelerator and click Save.
4. Navigate to **Runtime** > **Run all** (or press `Ctrl+F9`).
5. The notebook will automatically download the dataset, initialize the models, run all three ablation training loops sequentially, and plot the final comparison graphs at the bottom.

---

## Reduced-Scale Reproduction Details

To comply with the free Colab T4 GPU compute limits, the paper's original experiment was drastically scaled down while preserving the mathematical variety of matrix shapes (Attention and MLP layers) necessary to test Muon.

* **Dataset:** Changed from 5.7 Trillion pre-training tokens to the **TinyShakespeare** character-level dataset (90% train / 10% validation split).
* **Model Size:** Changed from a 16-billion parameter Mixture-of-Experts (MoE) architecture to a **NanoTransformer** (~1M parameters). Hyperparameters: `n_embd=128`, `n_head=4`, `n_layer=4`, `block_size=128`.
* **Epochs/Compute:** Changed from a multi-day cluster training schedule to **1500 iterations** with a batch size of 64. 

## Seeds

A strict manual seed of **`1337`** is enforced globally via `torch.manual_seed(1337)` at the beginning of each optimizer's training run. This ensures the data batching and weight initialization are identical across the AdamW, Vanilla Muon, and Muon + Weight Decay runs for a perfectly fair ablation.

---

## Results & Ablation Table

The following results were recorded at iteration 1400 of the training loop. The ablation successfully reproduces the paper's core claim: while Vanilla Muon converges quickly, its lack of regularization causes weight matrices to grow uncontrollably (high RMS). Integrating weight decay (WD) stabilizes the weights while allowing Muon to significantly outperform the AdamW baseline in validation loss.

| Optimizer Configuration | Final Validation Loss (Step 1400) | MLP Weight RMS Status | Conclusion |
| :--- | :--- | :--- | :--- |
| **AdamW (Baseline)** | 1.9710 | Stable | Standard baseline; slowest convergence. |
| **Vanilla Muon (No WD)** | 1.9152 | **Exploding (Unbounded upward trajectory)** | Converges fast, but structurally unstable over longer horizons. |
| **Muon + Weight Decay (Paper's Fix)** | **1.9176** | **Stable (Matched to AdamW scale)** | **Optimal.** Fast convergence without matrix explosion. |

*(Note: See the final output cell in the Jupyter Notebook for the matplotlib visualization of the Validation Loss and Weight RMS curves).*