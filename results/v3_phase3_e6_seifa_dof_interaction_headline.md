# v3.0 Phase 3 #3 — SEIFA × day-of-fortnight interaction

Single-experiment headline for `e6_seifa_dof_interaction`. Tests the Reading-C2 hypothesis from `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md`: does an explicit `sa2_seifa_x_dof` feature unlock the Centrelink-fortnightly-cycle × SEIFA-disadvantage interaction that v2.x's SHAP-interaction probe missed?

Full merged report: `results\v3_phase3_e6_seifa_dof_interaction_kfold.md`

## Headline

- **Mean Δ MAE: +0.670 c/L** (negative = Model B with interaction beats Model A)
- Stdev across 6 folds: 0.684 c/L
- Min: -0.107
- Max: +2.001
- **Verdict** (|Mean|>2×Stdev = robust): noise

## Compared to published PR B baseline (no interaction)

| Metric | This run (with interaction) | PR B baseline |
|--------|-----------------------------:|--------------:|
| Mean Δ MAE | +0.670 | +0.215 |
| Stdev | 0.684 | 0.394 |
| Min | -0.107 | -0.135 |
| Max | +2.001 | +1.042 |

## Per-fold Δ MAE (with interaction)

| Fold | MAE A | MAE B | Δ MAE |
|------|------:|------:|------:|
| fold_1 | 6.219 | 6.112 | -0.107 |
| fold_2 | 8.799 | 9.697 | +0.897 |
| fold_3 | 13.331 | 14.071 | +0.740 |
| fold_4 | 6.954 | 7.277 | +0.323 |
| fold_5 | 4.181 | 4.348 | +0.167 |
| fold_6 | 9.573 | 11.574 | +2.001 |

## Where does `sa2_seifa_x_dof` rank in Model B's gain importance?

Per-fold LightGBM `gain` importance for the new column + its rank among all Model B features (rank 1 = highest gain). High rank (top 20) means the model is actively using the interaction; low rank (bottom 25) means LightGBM found nothing.

| Fold | Gain | Rank (of all Model B features) |
|------|-----:|-------------------------------:|
| fold_1 | 16,965 | 46 |
| fold_2 | 40,110 | 48 |
| fold_3 | 5,773 | 56 |
| fold_4 | 18,140 | 47 |
| fold_5 | 8,890 | 58 |
| fold_6 | 13,907 | 54 |

## Reading

**Inconclusive — partial effect.** The interaction moves the headline but not robustly. Check the per-fold table above for where it helps vs hurts; if fold_3 or fold_6 specifically benefits, that's a Reading-C-leaning signal worth chasing.

## Sources

- `results\v3_phase3_e6_seifa_dof_interaction_kfold.md` — full per-fold report
- `tools/research/v3_phase3_interaction_experiment.py` — this script
- `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md` — Reading C2 hypothesis + ranked next-steps
- `results/v3_phase2_pr_b_baseline_kfold.md` — reference numbers
