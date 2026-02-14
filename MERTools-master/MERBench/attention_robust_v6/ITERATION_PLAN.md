# Attention Robust V6 Iteration Plan

## 1. Baseline Lock (Current Best Version)
- Source: `MERTools-master/MERBench_Run/attention_robust_v2/results_comparison.md`
- Best balanced model (avg test ACC): `P-RMF V2 (VAE)`
  - test1 ACC: `0.8345`
  - test2 ACC: `0.7718`
  - test3 ACC: `0.9029`

## 2. Detailed Audit (Architecture / Training / Params)
### 2.1 Architecture audit
- `v2` issue: modality dropout is applied to sampled `z` but fusion uses `mu`, so dropout does not affect fusion path.
  - Ref: `toolkit/models/attention_robust_v2.py:219-227`
- `v5` issue: `mixup_data()` exists but is not connected to training forward path, so mixup is effectively inactive.
  - Ref: `toolkit/models/attention_robust_v5.py:362-395` and forward path `toolkit/models/attention_robust_v5.py:483-537`

### 2.2 Training logic audit
- Training objective: `CE + MSE + interloss(VAE regularizers)` from `main-robust.py`.
  - Ref: `main-robust.py:113-121`
- Model selection uses eval metric `emoval = F1 - 0.25 * MSE`.
  - Ref: `toolkit/utils/metric.py:9-21`, `main-robust.py:334-360`

### 2.3 Parameter audit (best-known v2 family)
- Typical stable settings from existing runs:
  - `hidden_dim=128`, `dropout=0.35`
  - `kl_weight=0.01`, `recon_weight=0.1`, `cross_kl_weight=0.01`
  - `modality_dropout=0.15`, `modality_dropout_warmup=20`
  - `lr=5e-4`, `l2=5e-5`, `epochs=100`

## 3. V6 Implementation Scope
- Base on `v2` to keep strongest overall performance profile.
- Add only verified, low-risk changes:
  1. Apply modality dropout directly to `mu` used by fusion.
  2. Set dropped modalities to very large uncertainty before fusion.
  3. Keep reconstruction on original latent `z` to avoid conflict with dropout path.
  4. Add dynamic KL warmup (`use_dynamic_kl`, `kl_warmup_epochs`).

## 4. V6 Training Start
- Script: `attention_robust_v6/train_v6.sh`
- Model name: `attention_robust_v6`
- Initial run target: one full training with best v2-style hyperparameters + V6 fixes.

## 5. Next Iteration Candidates (after first V6 run)
1. `fusion_temperature`: `[0.8, 1.0, 1.2]`
2. `modality_dropout`: `[0.10, 0.15]`
3. `kl_warmup_epochs`: `[10, 20, 30]`
