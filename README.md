# Joint Multimodal Diffusion Model for Multi-Horizon Probabilistic PV Forecasting

Research codebase and single master executable notebook extending the **PV-MM-Diffusion** architecture (*Huang et al., Applied Energy 2026*) to multi-horizon solar forecasting (15, 30, and 60 minutes ahead) on the Stanford SKIPP'D dataset.

---

## 🚀 Key Master File

All research logic, model components, training loops, ablations, baselines, and visualization suites have been implemented in:
- [`joint_multimodal_diffusion_solar_forecasting.ipynb`](file:///home/haseebumer/diffusion_attention_skippd_solar_forecasting/joint_multimodal_diffusion_solar_forecasting.ipynb)

---

## 🎯 Project Overview & Research Question

> **Core Research Question:** Does the benefit of joint sky-image and PV-power generation (versus a decoupled two-stage "predict future sky images $\to$ map images to PV power" pipeline) persist as the forecast horizon extends to 30 and 60 minutes?

### Methodological Architecture
1. **Coupled U-Net Backbone**:
   - **PV Branch**: 1D temporal convolutions (`tf.keras.layers.Conv1D`) + temporal Multi-Head Self-Attention (`MultiHeadAttention`), no downsampling along time, expanding channel capacity with depth across 4 hierarchical scales.
   - **Sky Image Branch**: Factorized spatio-temporal convolutions (spatial 2D convs + temporal 1D convs, MM-Diffusion video paradigm) + spatial and temporal self-attention across 4 U-Net scales with spatial downsampling ($64 \to 32 \to 16 \to 8$).
2. **Cross-Modal Fusion via RS-PV-MMA**:
   - **Random-Shift PV Multi-Modal Attention**: Custom Keras layer providing local temporal window cross-attention between PV representations and sky video features.
   - Random temporal offset sampled per forward pass during training via `tf.random` (compatible with both eager mode and `tf.function` graph mode), accommodating cloud-advection physical delays.
   - Configurable window sizes per scale: default `[1, 4, 8]` for 15-min, and adapted `[2, 6, 12]` / `[2, 8, 16]` for 30-min and 60-min horizons.
3. **Empty-Frame Placeholder Scheme**:
   - 3 operational input modes trained with equal mixed-mode loss weighting ($w_{SP}=1/3, w_{SI}=1/3, w_{PV}=1/3$):
     - **SI PV $\to$ SI PV**: Multimodal input $\to$ joint multimodal output.
     - **SI $\to$ SI PV**: Sky image only input $\to$ joint multimodal output (past PV zeroed).
     - **PV $\to$ PV**: PV power only input $\to$ PV power output (sky images masked).
4. **Diffusion Process & Fast Sampler**:
   - Discrete $T=1000$ DDPM forward process with linear noise schedule and $\epsilon$-prediction objective.
   - Custom `train_step` computing weighted MSE on future noise predictions for both modalities.
   - **DPM-Solver++** fast high-order ODE solver for high-fidelity probabilistic sampling in 15–60 steps instead of 1000.
5. **Multi-Horizon Paradigms**:
   - **Option A (MVP)**: Separate specialized models for 15-min (8 future frames), 30-min (15 future frames), and 60-min (30 future frames) at 2-minute cadence.
   - **Option B (Shared Model)**: Single unified Coupled U-Net conditioned on a horizon embedding flag ($h \in \{15, 30, 60\}$).

---

## 📊 Dataset & Setup

- **Dataset**: Stanford SKIPP'D dataset (30 kW rooftop array + $64 \times 64$ RGB sky camera images), preprocessed following Nie et al. (*SkyGPT*, 2024).
- **Extracted Location**: `/home/haseebumer/content/data/extracted/` (or `/content/data/extracted/` with automatic synthetic fallback if missing).
  - `trainval_images_log.npy`: `(349372, 64, 64, 3)` uint8
  - `trainval_pv_log.npy`: `(349372,)` float64
  - `times_trainval.npy`: `(349372,)` datetime
  - `test_images_log.npy`: `(14003, 64, 64, 3)` uint8
  - `test_pv_log.npy`: `(14003,)` float64
  - `times_test.npy`: `(14003,)` datetime
- **Data Pipeline**: Memory-mapped streaming `tf.data.Dataset` with day-boundary integrity, 2-min cadence (stride 2 on 1-min records), `.cache()`, `.shuffle()`, `.batch()`, `.prefetch()`.

---

## 🛠️ Training Infrastructure

- **Checkpointing with Automatic Resume**:
  - Uses `tf.train.Checkpoint` + `tf.train.CheckpointManager` saving after **every single epoch**.
  - Automatically detects the latest checkpoint on startup and restores weights, optimizer state, epoch, and step count.
  - Maintains a rolling window of the last $N=5$ checkpoints + a separate permanent "best" checkpoint (lowest validation loss).
- **Callbacks**:
  - `EarlyStopping` (configurable patience, restore best weights).
  - `TensorBoard` logging (scalars, sample prediction intervals, image grids).
  - `ResourceMonitorCallback` tracking GPU memory (VRAM) and epoch wall-clock duration.

---

## 📈 Evaluation Metrics & Baselines

- **CRPS** (Continuous Ranked Probability Score): Vectorized empirical predictive distribution formulation.
- **Winkler Score ($WS_{90}$)**: Evaluated at 90% nominal coverage ($\alpha = 0.10$, 5th and 95th percentiles).
- **Forecast Skill ($FS$)**: Relative to Smart Persistence Model baseline.
- **VGG-16 Cosine Similarity ($VGG\text{-}CS$)**: Feature similarity of predicted sky images using penultimate max-pool layer.
- **Baselines**:
  1. **Smart Persistence Model (SPM)**: Clear-sky index method ($k_{clr}(t) = P(t)/P_{clr}(t)$) with solar geometry.
  2. **Two-Stage Baseline**: Spatio-temporal video prediction model $\to$ PV regression model.
  3. **PV $\to$ PV Only Ablation**: Evaluated using the empty-frame placeholder.
