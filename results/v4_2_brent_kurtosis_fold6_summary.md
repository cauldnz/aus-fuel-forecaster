# v4.2 — Brent higher-moment (kurtosis + skew) features for fold_6

Nested test on top of v4.1 (realized vol). Adds rolling skew (30d) and kurtosis (30d, 60d) of Brent returns. Three-way comparison isolates the **marginal** value of higher moments beyond vol.

New vs v4.1: `upstream_brent_return_skew_30d`, `upstream_brent_return_kurt_30d`, `upstream_brent_return_kurt_60d`

## fold_6 seed-stdev — the three-way comparison

_All stdevs over the same 6 seeds ([1, 7, 13, 42, 99, 123]) — matched-denominator. Trustworthy only at the full 6-seed sweep._

| Config | fold_6 seed-stdev |
|--------|------------------:|
| Clean baseline (no extra feats) | 0.1002 |
| v4.1 (+ vol) | 0.0506 |
| v4.2 (+ vol + skew + kurtosis) | 0.0775 |

**Marginal verdict (v4.2 vs v4.1): HIGHER MOMENTS HURT (beyond vol)**

## Per-fold seed-stdev — all configs

| Fold | Clean baseline | v4.1 vol-only | v4.2 vol+moments |
|------|---------------:|--------------:|-----------------:|
| fold_1 | 0.0191 | 0.0316 | 0.0332 |
| fold_2 | 0.0307 | 0.0187 | 0.0065 |
| fold_3 | 0.0436 | 0.0832 | 0.0452 |
| fold_4 | 0.0398 | 0.0395 | 0.0751 |
| fold_5 | 0.0454 | 0.0499 | 0.0506 |
| fold_6 | 0.1002 | 0.0506 | 0.0775 |

## Per-fold seed-mean MAE — v4.2 vs clean baseline

| Fold | Clean baseline | v4.2 | Δ mean |
|------|---------------:|-----:|-------:|
| fold_1 | 5.3335 | 5.3275 | -0.0060 |
| fold_2 | 4.7624 | 4.8200 | +0.0576 |
| fold_3 | 6.0898 | 6.0479 | -0.0419 |
| fold_4 | 5.0015 | 5.0770 | +0.0755 |
| fold_5 | 4.4115 | 4.3794 | -0.0322 |
| fold_6 | 3.5453 | 3.5863 | +0.0410 |

## Per-seed per-fold MAE_A (v4.2)

| Seed | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | fold_6 | Wall-clock |
|------|---|---|---|---|---|---|----|
| 42 | 5.3646 | 4.8086 | 5.9735 | 5.0604 | 4.3638 | 3.5735 | 0.0 min (cached) |
| 1 | 5.2838 | 4.8307 | 6.1146 | 5.0596 | 4.3722 | 3.5196 | 0.0 min (cached) |
| 7 | 5.3115 | 4.8175 | 6.0227 | 5.2022 | 4.3966 | 3.5490 | 0.0 min (cached) |
| 13 | 5.3748 | 4.8221 | 6.0300 | 5.1460 | 4.2933 | 3.6825 | 0.0 min (cached) |
| 99 | 5.3317 | 4.8209 | 6.0720 | 4.9978 | 4.3853 | 3.4954 | 0.0 min (cached) |
| 123 | 5.2986 | 4.8200 | 6.0748 | 4.9960 | 4.4649 | 3.6981 | 0.0 min (cached) |

## Sources

- `tools/research/v4_2_brent_kurtosis_fold6.py` — this script
- `results/v4_1_brent_vol_fold6.json` — v4.1 (vol-only) baseline for the marginal comparison
- `results/v3_phase3_hyperopt_validation.json` — clean baseline
- User domain pointer: kurtosis of returns as a regime feature
