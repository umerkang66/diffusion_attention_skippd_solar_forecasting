# Joint Multimodal Diffusion Model for Multi-Horizon Solar Forecasting

An implementation in TensorFlow 2.x / Keras extending joint multimodal diffusion forecasting ([PV-MM-Diffusion](https://doi.org/10.1016/j.apenergy.2025.125000)) to multi-horizon solar forecasting (15, 30, and 60 minutes ahead) on the Stanford SKIPP'D benchmark dataset.

Instead of a decoupled two-stage pipeline (predicting future sky images first, then mapping images to power via a separate regression model), this framework jointly denoises future sky camera frames and future photovoltaic (PV) power sequences within a single generative reverse diffusion process.

---

## Architecture Overview

![System Architecture](architecture.png)

- **Dual-Stream Backbone**: A 1D temporal convolution stream for PV power paired with a factorized (2D spatial + 1D temporal) video stream for sky camera images across 4 resolution scales.
- **Cross-Modal Attention (RS-PV-MMA)**: Bidirectional cross-attention between sky video features and PV power sequences, featuring a random temporal shift during training to simulate cloud advection latency.
- **Empty-Frame Conditioning**: Supports inference under missing sensor conditions (full multimodal, sky-image only, or PV-only).
- **Fast Sampling (DPM-Solver++)**: High-order ODE solver that generates probabilistic predictions (30 Monte Carlo samples) in 15 to 60 steps rather than 1,000 discrete diffusion steps.

---

## Forecast Horizons

Historical context is fixed at 8 frames (16 minutes at a 2-minute sampling cadence):

| Horizon | Lead Time | History | Forecast Target | Total Sequence Length |
| :--- | :--- | :--- | :--- | :--- |
| **15-min** | 15 minutes | 8 frames (16 min) | 8 frames (16 min) | 16 frames |
| **30-min** | 30 minutes | 8 frames (16 min) | 15 frames (30 min) | 23 frames |
| **60-min** | 60 minutes | 8 frames (16 min) | 30 frames (60 min) | 38 frames |

Two modeling strategies are included:
1. **Option A (Specialized Models)**: Separate models trained independently for each horizon (15m, 30m, 60m).
2. **Option B (Shared Model)**: A single unified model conditioned on a horizon embedding.

---

## Repository Structure

```
.
├── README.md                                       # Project documentation
├── architecture.png                                # Architecture diagram
├── joint_multimodal_diffusion_solar_forecasting.ipynb  # Main end-to-end notebook
├── checkpoints/                                    # Model weights and resume checkpoints
└── PROMPT.md                                       # Research specifications
```

---

## Dataset & Preprocessing

The model uses the **Stanford SKIPP'D** (SKy Image Processing for Photovoltaic Determinism) dataset:
- **Sky Images**: Downsampled to 64x64 RGB at a 2-minute cadence.
- **PV Power**: 30 kW rooftop array, aligned to 2-minute intervals and normalized to [0, 1].
- **Filtering**: Clear-sky periods filtered out; only complex cloudy days retained.

### Expected Data Path

Place extracted `.npy` files in `/content/data/extracted/` (or configure your local path in the notebook):
- `trainval_images_log.npy`
- `trainval_pv_log.npy`
- `times_trainval.npy`
- `test_images_log.npy`
- `test_pv_log.npy`
- `times_test.npy`

> **Note**: If raw data files are not present, the notebook includes an automated synthetic data generator with matching tensor shapes and diurnal patterns for quick pipeline verification.

---

## Requirements & Setup

### Environment
- Python 3.9+
- TensorFlow 2.12+
- CUDA-compatible GPU (recommended)

### Installation
```bash
pip install tensorflow numpy pandas scipy matplotlib scikit-learn pvlib
```

### Hardware Recommendations
- **15-min model**: 8 GB GPU VRAM
- **30-min model**: 12 GB GPU VRAM
- **60-min model**: 16 GB GPU VRAM

---

## Usage

Open and run the master notebook:

```bash
jupyter lab joint_multimodal_diffusion_solar_forecasting.ipynb
```

### Notebook Workflow
1. **Data Pipeline**: Loads memory-mapped arrays, filters day boundaries, and creates `tf.data` pipelines.
2. **Model Definition**: Builds the Coupled U-Net backbone, cross-attention layers, and diffusion model.
3. **Training**: Auto-resumes from checkpoints, saves best weights, and logs metrics to TensorBoard.
4. **Evaluation**: Generates 30 Monte Carlo sample paths per test instance using DPM-Solver++ and computes evaluation metrics.
5. **Visualization**: Plots probabilistic forecast bands (P10, P50, P90) alongside generated sky video frames.

### TensorBoard Monitoring
```bash
tensorboard --logdir logs/
```

---

## Evaluation Metrics

Probabilistic solar forecasts are evaluated across 30 sample paths:
- **CRPS (Continuous Ranked Probability Score)**: Measures probabilistic accuracy and calibration.
- **Winkler Score (WS-90)**: Evaluates reliability and sharpness of the 90% prediction interval (P5 to P95).
- **Forecast Skill (FS)**: Percentage improvement in CRPS over a Smart Persistence Model (SPM).
- **VGG Cosine Similarity**: Evaluates visual quality and cloud pattern fidelity of generated sky frames.

---

## References

- **PV-MM-Diffusion**: Huang et al., *"PV-MM-Diffusion: A Multimodal Diffusion Model for Ultra-Short-Term Photovoltaic Power Forecasting Using Sky Images"*, Applied Energy, 2026.
- **SkyGPT**: Nie et al., *"SkyGPT: Probabilistic Sky Image Prediction for Solar Forecasting"*, Applied Energy, 2024.
- **DPM-Solver++**: Lu et al., *"DPM-Solver++: Fast Solver for Guided Samplers of Diffusion Probabilistic Models"*, NeurIPS, 2022.
- **DDPM**: Ho et al., *"Denoising Diffusion Probabilistic Models"*, NeurIPS, 2020.
