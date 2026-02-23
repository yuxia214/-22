# AttentionRobust V11 Experiment Pack

This folder is a non-invasive experiment pack built on top of `attention_robust_v2`.
It does not modify your original v2 files.

## Goal

Improve accuracy with a reproducible and staged workflow:

1. Stage smoke: one-run pipeline check.
2. Stage short: small candidate screening.
3. Stage formal: 6-repeat evaluation for stable mean/std.

## Dataset Metric Mapping

The training output names are mapped as:

- `cv` -> `Train&Val`
- `test1` -> `MER-MULTI`
- `test2` -> `MER-NOISE`
- `test3` -> `MER-SEMI` (WAF only)

Reported summary columns:

- `Train&Val WAF/MSE`
- `MER-MULTI WAF/MSE`
- `MER-NOISE WAF/MSE`
- `MER-SEMI WAF`
- `Average WAF` and `Average MSE` (`Train&Val + MULTI + NOISE`)

## Files

- `launch_main_robust_seeded.py`: seeded launcher for `main-robust.py` (no core code edit).
- `train_v11_smoke.sh`: one-run sanity check.
- `train_v11_short.sh`: short candidate screening (default 2 seeds).
- `train_v11_formal.sh`: formal repeated runs (default 6 seeds).
- `train_v11_formal_from_short.sh`: auto-select top-k from short summary, then formal repeats.
- `train_v11_group_d.sh`: extra group D (high-capacity, hidden_dim=256).
- `train_v11_group_e.sh`: extra group E (parallel-friendly, smaller batch).
- `train_v11_group_f.sh`: extra group F (CPU-lean, low thread + light workers).
- `train_v11_group_g.sh`: extra group G (DataLoader overlap, workers=2).
- `summarize_v11.py`: aggregate run outputs into table-friendly CSV.
- `start_v11_screen.sh`: run any stage in screen.
- `start_v11_dual_screen.sh`: start group D + E concurrently.
- `start_v11_dual_fg_screen.sh`: start group F + G concurrently.

## Usage

From project root:

```bash
bash attention_robust_v11/train_v11_smoke.sh
bash attention_robust_v11/train_v11_short.sh
REPEATS=6 bash attention_robust_v11/train_v11_formal.sh
```

Screen mode:

```bash
bash attention_robust_v11/start_v11_screen.sh smoke
bash attention_robust_v11/start_v11_screen.sh short
bash attention_robust_v11/start_v11_screen.sh formal
bash attention_robust_v11/start_v11_screen.sh promote
bash attention_robust_v11/start_v11_screen.sh group_d
bash attention_robust_v11/start_v11_screen.sh group_e
bash attention_robust_v11/start_v11_screen.sh group_f
bash attention_robust_v11/start_v11_screen.sh group_g
```

Dual concurrent start:

```bash
bash attention_robust_v11/start_v11_dual_screen.sh
# or choose GPU ids explicitly
GPU_A=0 GPU_B=0 bash attention_robust_v11/start_v11_dual_screen.sh
# dedicated launcher for new F/G groups
bash attention_robust_v11/start_v11_dual_fg_screen.sh
```

CPU tuning knobs (all stages):

```bash
THREADS_PER_RUN=2 CPU_CORES=0-5 bash attention_robust_v11/train_v11_short.sh
# control dataloader workers too
THREADS_PER_RUN=1 NUM_WORKERS=2 CPU_CORES=6-11 bash attention_robust_v11/train_v11_group_g.sh
```

Top-k formal promotion:

```bash
# 1) quick pre-screen
SHORT_EPOCHS=30 REPEATS=2 bash attention_robust_v11/train_v11_short.sh
# 2) promote top-2 by balanced score, run formal repeats
TOPK=2 RANK_KEY=score_balanced_mean REPEATS=6 bash attention_robust_v11/train_v11_formal_from_short.sh
```

## Output

- Results: `attention_robust_v11/outputs/results`
- Logs: `attention_robust_v11/outputs/logs`
- Summary CSV: `attention_robust_v11/outputs/summary`
