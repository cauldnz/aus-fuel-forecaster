# v6.1 — PC-compressed augmentor block (4 PCs) vs Model A

Distils the full 37-col augmentor surface to its top 4 principal components (the v6 intrinsic dimensionality, ~80% variance) and tests Model A vs Model B-PC under 6-fold k-fold. The augmentor's best possible representation — orthogonal, low-rank, overfit-resistant.

Full report: `results\v6_1_pc_model_b_kfold.md`

## Headline

- **Mean Δ MAE: -0.005 c/L** (negative = Model B-PC beats Model A)
- Stdev across 6 folds: 0.233
- **Verdict: noise**

## vs the raw-column augmentor (v3.0 t+1 PR B baseline)

| Config | Mean Δ MAE | Stdev |
|--------|-----------:|------:|
| Raw 15-col SA2 block | +0.215 | 0.394 |
| 4-PC compressed block | -0.005 | 0.233 |

## Per-fold Δ MAE

| Fold | MAE A | MAE B-PC | Δ MAE |
|------|------:|---------:|------:|
| fold_1 | 5.854 | 5.979 | +0.124 |
| fold_2 | 8.527 | 8.095 | -0.432 |
| fold_3 | 12.219 | 12.460 | +0.241 |
| fold_4 | 6.126 | 6.185 | +0.059 |
| fold_5 | 3.897 | 4.060 | +0.163 |
| fold_6 | 8.976 | 8.789 | -0.187 |

## Reading

**Null — the PC block is no better than the raw block.** Even the augmentor's information distilled to its 4 orthogonal essence-dimensions adds nothing over Model A. This is the final nail: the null isn't a representation artifact (raw vs PC), it's the information content. The ~4 socioeconomic gradients are redundant with what the lag features already encode.

## Sources

- `results\v6_1_pc_model_b_kfold.md` — full per-fold report
- `tools/research/v6_1_pc_model_b.py` — this script
- `docs/research/2026-06_v6_augmentor_intrinsic_dimensionality.md` — the PCA analysis
- `results/v3_phase2_pr_b_baseline_kfold.md` — raw-column reference
