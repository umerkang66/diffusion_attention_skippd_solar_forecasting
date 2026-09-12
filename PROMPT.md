PROJECT: Joint Multimodal Diffusion Model for Multi-Horizon Probabilistic PV Forecasting

## Goal
Implement a research codebase in TensorFlow (TF2.x / Keras) that extends the 
PV-MM-Diffusion architecture (a single coupled DDPM that jointly generates future 
sky images AND future PV power sequences in one denoising process) from a fixed 
15-minute forecast horizon to THREE forecast horizons: 15, 30, and 60 minutes ahead 
— while keeping the joint (non-two-stage) architecture intact. The research question: 
does the benefit of joint image+power generation (vs. two-stage "predict images then 
map to power") persist as the forecast horizon grows?

## Dataset

### DOWNLOADED DATASE INFORMATION
Extracted Data is present at '/content/data/extracted/'

In the previous folder these are the files:

test_images_log.npy
test_pv_log.npy
times_test.npy
times_trainval.npy
trainval_images_log.npy
trainval_pv_log.npy


Use the SKIPP'D dataset (sky images + PV power from a 30kW rooftop system at Stanford 
University), as preprocessed/partitioned by the SkyGPT paper (Nie et al. 2024).
- Source repo for preprocessing reference: https://github.com/yuhao-nie/SkyGPT
  (also check https://github.com/yuhao-nie/SKIPPD for the raw dataset and loader)
- Images: 64x64 RGB, downsampled from 2048x2048 fisheye camera
- PV power: aligned at 1-min native resolution, resampled to 2-min intervals
- Only multi-cloud conditions retained (clear-sky days filtered out)
- If direct download isn't available, write a tf.data-based loading module with 
  clear instructions for where to place the data locally, and mock/synthetic data 
  generators so the pipeline can be tested end-to-end without the real dataset.
- Build the data pipeline using tf.data.Dataset (with .cache(), .shuffle(), 
  .batch(), .prefetch()) for efficient loading of paired image+power sequences.

Set up train/val/test splits following the original SkyGPT convention (~88/7/4%), 
and clearly document any deviation.

## Multi-horizon design (core novelty — implement BOTH, start with Option A)

Historical context: fixed at 8 frames = 16 minutes of history (2-min cadence), 
for both sky images and PV sequence, matching the original paper.

Option A (MVP — implement first):
Train three separate model instances, one per horizon, by extending only the
FUTURE sequence length:
  - 15-min horizon: 8 future frames (2-min cadence)
  - 30-min horizon: 15 future frames
  - 60-min horizon: 30 future frames
Each model is otherwise architecturally identical (same Coupled U-Net + cross-modal
attention). This isolates "how does joint generation degrade/scale with horizon 
length" as a clean, controlled comparison.

Option B (stretch goal, implement after A works):
A single shared model conditioned on a horizon embedding/flag, trained jointly across
all three horizons, to test whether one network can generalize across horizons 
without separate training (more novel, more engineering risk — treat as secondary 
experiment).

## Architecture (replicate from PV-MM-Diffusion, then extend) — build as tf.keras.Model subclasses

1. **Backbone**: Coupled U-Net with two branches:
   - PV branch: 1D convolutions (tf.keras.layers.Conv1D) + temporal self-attention 
     (tf.keras.layers.MultiHeadAttention), no downsampling along time, channels 
     increase with depth
   - Sky image branch: spatial 2D convs (Conv2D) + temporal 1D convs (factorized, 
     not full 3D conv) + spatial & temporal attention (follow MM-Diffusion's approach 
     for video)
   - 4 U-Net scales; cross-modal attention introduced at scales [2,3,4]

2. **Cross-modal fusion**: Implement Random-Shift PV Multi-Modal Attention (RS-PV-MMA)
   as a custom Keras layer:
   - Local temporal window cross-attention (not global) between PV segments and
     sky-image segments, with a randomly sampled temporal offset each forward pass
     (use tf.random for the offset, ensure it works correctly under tf.function/
     graph mode — test eagerly first, then verify graph-mode compatibility)
   - Default window sizes per scale: [1, 4, 8] (top to bottom) — but re-tune this
     per horizon, since window sizes were only validated for the 15-min case; note
     this explicitly as a design decision to test in ablation for the 30/60-min 
     models

3. **Empty-frame placeholder scheme**: Support three input modes via zero-tensor
   placeholders (train with mixed-mode loss, default equal weighting 1/3 each):
   - "SI PV → SI PV" (both modalities in, both out)
   - "SI → SI PV" (images only in)
   - "PV → PV" (power only in/out)

4. **Diffusion process**: Standard DDPM, implemented as custom training logic 
   (use a custom train_step in a tf.keras.Model subclass, or a manual training loop 
   with tf.GradientTape — whichever makes the mixed-mode loss and diffusion-step 
   sampling cleanest):
   - T = 1000 diffusion steps, linear noise schedule
   - ε-prediction objective (predict added noise, not clean data)
   - Loss: MSE on noise prediction for both modalities' future segments, weighted
     by mixed-mode loss weights (w_SP, w_SI, w_PV = 1/3 each by default)
   - Sampling: implement DPM-Solver++ for fast inference (60 solver steps as 
     default, but re-validate this step count for the 30/60-min horizons since 
     longer sequences may need different step counts — run a step-count sensitivity 
     sweep per horizon, mirroring the original paper's Table 7/Fig. 10 approach)

## Training infrastructure requirements (important — implement carefully)

1. **Checkpointing with automatic resume**:
   - Use tf.train.Checkpoint + tf.train.CheckpointManager to save a checkpoint 
     after EVERY epoch (not just best-so-far), including model weights, optimizer 
     state, current epoch number, and current diffusion training step count.
   - On training script startup, always check for an existing checkpoint in the 
     designated checkpoint directory and automatically restore from the latest one 
     if found, then resume training from that exact epoch — training should be 
     fully resumable after an interruption (crash, timeout, manual stop) with no 
     manual intervention needed.
   - Keep a rolling window of the last N checkpoints (e.g., N=5) plus a separate 
     "best" checkpoint (lowest validation CRPS/loss) that is never overwritten 
     by the rolling window.
   - Log clearly at startup whether training is starting fresh or resuming, and 
     from which epoch.

2. **Callbacks** (implement as tf.keras.callbacks.Callback subclasses, or use 
   built-in ones where they fit):
   - **EarlyStopping**: monitor validation loss (or validation CRPS if you compute 
     it during validation), with a configurable patience (default ~15-20 epochs 
     given how slow diffusion training/sampling can be), restore_best_weights=True
   - **TensorBoard**: log training loss, validation loss, learning rate, and 
     (periodically, e.g. every N epochs) sample-quality metrics (VGG-CS on a few 
     validation samples, a few generated sky-image grids, a few sampled PV 
     forecast plots with prediction intervals) as images/scalars in TensorBoard
   - **ModelCheckpoint**-equivalent (can be custom given the CheckpointManager 
     above, or combined with it) to also save the single best model separately 
     from the rolling per-epoch checkpoints
   - **ReduceLROnPlateau** or a custom LR schedule callback (optional but useful 
     given multi-day training runs)
   - A custom callback to log GPU memory usage and epoch wall-clock time, useful 
     for later reporting training cost in the paper

3. Make sure all of the above works correctly whether you use a custom training 
   loop (tf.GradientTape) or model.fit() — if diffusion training needs enough 
   custom logic that model.fit() doesn't cleanly fit, implement an equivalent 
   custom loop that manually replicates checkpointing/early-stopping/tensorboard 
   logging behavior (don't skip these just because a custom loop is being used).

## Evaluation

Implement these metrics exactly as formulas (don't just call an off-the-shelf 
library without verifying the formula matches):
- **CRPS** (Continuous Ranked Probability Score) — via the standard closed-form or
  empirical-CDF approximation, averaged over test samples
- **Winkler Score (WS)** at 90% nominal coverage (5th/95th percentile bounds),
  formula: WS_k = δ if in-interval; δ + 2(L-x)/α if below; δ + 2(x-U)/α if above,
  where δ = U-L, α=0.1
- **Forecast Skill (FS)** relative to the Smart Persistence Model baseline (implement
  SPM using the clear-sky index method: k_clr(t) = P(t)/P_clr(t), forecast =
  k_clr(t) × P_clr(t+T), with P_clr from a clear-sky solar geometry model)
- **VGG Cosine Similarity (VGG CS)** for sky-image quality, using 
  tf.keras.applications.VGG16 (pretrained on ImageNet, include_top=False), 
  penultimate max-pool features, resizing 64x64 frames to 224x224 first
- Report all metrics SEPARATELY per horizon (15/30/60 min) in a comparison table,
  plus generate the accuracy-vs-horizon and WS-vs-horizon trend plots (matplotlib, 
  similar to Fig. 7/9 style in the reference paper)
- Use 30 stochastic samples per input to form the empirical predictive distribution
  at test time (matching the original protocol); also report a reduced-sample
  (10-sample) deployment-friendly configuration with latency numbers

## Baselines to implement for comparison

1. Smart Persistence Model (SPM) — deterministic, closed-form
2. A simple two-stage baseline: separately trained video-prediction model → PV
   regression model (to directly test the "joint vs. two-stage" hypothesis at
   each horizon — this is the most important baseline for the paper's argument)
3. PV→PV only mode (ablation, using your own model's empty-frame placeholder)

## Literature comparison (new requirement)

In addition to your own baselines, produce a "results-in-context" comparison table 
that places your model's numbers alongside PUBLISHED results from these papers 
(search for and cite the actual reported numbers — do not fabricate any figures; 
if a paper doesn't report a metric/horizon you need, mark it "not reported" rather 
than estimating):
  - PV-MM-Diffusion (Huang et al., Applied Energy 2026) — CRPS=2.63kW, WS=21.46kW 
    at 15-min horizon (SI PV → SI PV mode)
  - SkyGPT → U-Net (Nie et al. 2024) — CRPS=2.81kW at 15-min horizon
  - Rastgoo et al., "A Diffusion-Based Probabilistic Ultra-Short-Term Solar Power 
    Prediction Using the Sky Image Sequences" (IEEE Access) — compares Diffusion 
    vs VAE vs VGAN vs WGAN, includes a 60-min horizon
  - "Multimodal ultra-short-term probabilistic solar power forecasting with 
    generative AI and Transformer" (Applied Energy, 2025) — reports CRPS and MWIS 
    at 15/30/60-min horizons (different dataset — Kuitun; flag this clearly as a 
    caveat, not a like-for-like comparison, since site/system differences affect 
    absolute kW-scale metrics)
  - Any other directly relevant papers found via web search at the time of writing 
    up results (search Google Scholar / arXiv for recent papers on "diffusion 
    photovoltaic forecasting sky images multi-horizon" and "SKIPP'D multimodal 
    forecasting" before finalizing this table, since new papers may have appeared)
Present this as a clearly labeled table with a footnote explaining that cross-paper 
comparisons are confounded by different datasets/sites/systems, and that your own 
apples-to-apples baselines (SPM, two-stage, PV→PV) are the primary basis for claims, 
while the literature table is contextual framing only.

## Deliverables

1. `data/` — tf.data pipeline, preprocessing, and multi-horizon windowing code
2. `models/` — Coupled U-Net, RS-PV-MMA attention layer, diffusion process
   (forward/reverse), DPM-Solver++ sampler, all as Keras layers/models
3. `train.py` — training loop (custom loop or model.fit with custom train_step) 
   with config files (YAML/JSON) per horizon, mixed-mode loss, checkpoint-per-epoch 
   with auto-resume, EarlyStopping, TensorBoard, and other callbacks described above
4. `evaluate.py` — computes CRPS, WS, FS, VGG-CS on test set, generates comparison
   tables (CSV/markdown) and plots, including the literature comparison table
5. `configs/` — one config per horizon (15/30/60 min) with clearly documented
   hyperparameters (sequence lengths, window sizes, solver steps, checkpoint dir)
6. `experiments/` — scripts to reproduce the ablations: window-size sensitivity
   per horizon, solver-step sensitivity per horizon, joint-vs-two-stage comparison
7. `README.md` — clear setup instructions, dataset download/placement instructions,
   how to run training/eval for each horizon, how checkpoint-resume works, how to 
   launch TensorBoard, and expected runtime/compute requirements (be honest about 
   GPU memory/time estimates, especially for the 60-min horizon which has ~4x 
   longer sequences than the 15-min case)
8. A results summary document (markdown) that reports final CRPS/WS/FS numbers
   per horizon, the joint-vs-two-stage comparison, AND the literature comparison 
   table, written in a style suitable for adapting into a paper's Results section

## Constraints & instructions for implementation

- Use TensorFlow 2.x with Keras. Keep the codebase modular so swapping the 
  diffusion sampler, attention window sizes, or horizon config doesn't require 
  rewriting the model.
- Start with a small/fast configuration (reduced U-Net width, fewer training
  steps, small subset of data) to get an end-to-end pipeline running and
  validated correctly BEFORE scaling up to full training — confirm shapes,
  losses decreasing, checkpoint save/resume working, and sampling working 
  correctly on a toy run first.
- Flag any point where you have to make a design decision not fully specified
  above (e.g., exact channel counts, batch size, learning rate schedule) and
  briefly justify the choice.
- Include unit tests for: the diffusion forward/reverse process, the
  cross-modal attention layer's shape handling across the three horizon configs, 
  and the checkpoint save/resume logic (simulate an interruption and confirm 
  training resumes from the correct epoch with matching loss trajectory).
- If GPU resources are limited, provide a clearly marked "reduced-scale" config
  (smaller image resolution, fewer channels, fewer diffusion steps) so the full
  pipeline can still be validated on modest hardware, with a note on what would
  need scaling up for publication-quality results.

Work through this in phases: (1) data pipeline, (2) core architecture + diffusion
process for the 15-min case (replication) with checkpointing/callbacks working 
end-to-end, (3) validate against reported numbers where possible, (4) extend to 
30/60-min horizons, (5) implement baselines, (6) run full evaluation and 
ablations, (7) build the literature comparison table, (8) write up results.

ALL OF THE WORLD SHOULD BE DONE IN an SINGLE .ipynb file