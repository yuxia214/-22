# AttentionRobust V7 Iteration Plan

## 1) Target (official metric alignment)
- MER-MULTI winner combined: 0.7005
- MER-NOISE winner combined: 0.6846
- MER-SEMI winner discrete(F1): 0.8911

Current best reference in repo (P-RMF V2):
- test1: F1 0.8348, MSE 0.6431 => combined 0.6740
- test2: F1 0.7832, MSE 0.5932 => combined 0.6349
- test3: F1 0.8995

Primary gaps:
- MULTI gap: -0.0265
- NOISE gap: -0.0497
- SEMI: already +0.0084 over winner F1

## 2) Design focus
Because combined = F1 - 0.25*MSE, V7 focuses on reducing MSE without sacrificing F1.

## 3) V7 architecture updates (over V6)
1. Emotion-guided valence prior fusion
- Compute emotion probability from logits.
- Map emotion probs to valence prior via learnable emotion-valence centers.
- Final valence = gated blend(raw regression, prior valence).

2. Emotion-valence consistency regularization
- Add SmoothL1 consistency between final valence prediction and prior valence.
- Add center regularization to keep emotion-valence centers stable.

3. Noise-oriented augmentation
- Add training-time feature Gaussian noise with warmup + probability control.
- Targets MER-NOISE robustness.

4. Keep V6 stable path
- mu-path modality dropout effective in fusion.
- dynamic KL warmup retained.

## 4) Training-logic updates
- Add weighted multitask loss:
  total = interloss + w_emo * CE + w_val * reg_loss
- Add selectable regression loss (`mse` / `smoothl1`).

## 5) Run strategy
- Start from one strong config emphasizing combined metric:
  - slightly larger valence weight
  - smoothl1 regression
  - moderate noise augmentation
- After this run, sweep:
  1) `val_loss_weight`: [1.2, 1.4]
  2) `valence_consistency_weight`: [0.08, 0.12]
  3) `feature_noise_std`: [0.02, 0.03]
