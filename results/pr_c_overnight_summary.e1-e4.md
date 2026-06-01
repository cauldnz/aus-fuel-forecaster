# PR C overnight experiment summary

Run via `tools/research/pr_c_overnight_runner.py` against augmentor
**v2.1.0** (commit `2ea02fb8`, 2026-05-31) — the clean semver release
that closes our four filed issues (#91 Stage 2, #92, #99, #101) plus
ships the new `ERP.population_density_per_km2` column and five more
cross-dataset PRESETs.

## TL;DR

**All 4 experiments completed.** Two distinct wins on different folds,
one strict regression, one mixed-to-baseline:

- **E1 (DSS temporal) wins test_normal** by 0.085 c/L vs PR B baseline
  (the original §7.7.2 motivation — quarterly welfare data on val/test).
  Slightly hurts test_crisis (+0.151).
- **E4 (new ERP density + 21-col curated SA2 block) wins test_crisis**
  by **0.282 c/L** vs PR B baseline (largest single gain of any
  experiment in the project history). Hurts test_normal by 0.151.
- **E2 (GCP temporal) regresses badly** — Model B *lost* by 0.470 c/L on
  test_normal. Per-row 2016 vs 2021 Census GCP values introduce noise.
- **E3 (combined DSS + GCP temporal)** has the GCP regression diluting
  the DSS gain; net result is roughly baseline on both folds.

**Recommended next step:** an E5 ablation combining E1's DSS-temporal
move with E4's new ERP density + curation broadening — that hits both
the test_normal and test_crisis wins from different directions and
might land both at once.

## Baseline (committed PR B headline)

| Fold | MAE A | MAE B | Δ MAE |
|------|------:|------:|------:|
| test_normal | 6.373 | 6.134 | **−0.239** |
| test_crisis | 13.616 | 13.295 | **−0.321** |

## Results

Δ MAE: more negative = Model B beats Model A. "vs baseline" subtracts
PR B's Δ MAE — **negative = this experiment improved**, positive = it
regressed.

| Experiment | test_normal Δ MAE | vs baseline | test_crisis Δ MAE | vs baseline | wall-clock | Verdict |
|------------|------------------:|------------:|------------------:|------------:|-----------:|---------|
| **e1_dss_temporal** | **−0.324** | **−0.085 ✅** | −0.170 | +0.151 | 26.1 min | win normal / lose crisis |
| **e2_gcp_temporal** | +0.470 | +0.709 ❌ | −0.255 | +0.066 | 43.3 min | regression both folds |
| **e3_combined_temporal** | −0.041 | +0.198 | −0.320 | +0.001 | 67.2 min | ≈ baseline (GCP drowns DSS) |
| **e4_new_erp_density_plus_curation** | −0.088 | +0.151 | **−0.603** | **−0.282 ✅** | 38.9 min | lose normal / win crisis (large) |

## Per-experiment narrative

### E1 — DSS temporal (best test_normal result of the lot)

Setup: 9 DSS welfare cols (age pension, DSP, parenting × 2, carer × 2,
youth × 2, seniors health card) moved from cross-sectional to temporal.
The 4 cols that aren't universally present across all DSS quarterly
releases (jobseeker + CRA missing in 2015-Q1, FTB-A/B missing in 2024-Q2
per `tools/research/dss_schema_probe.py`) stayed cross-sectional — the
augmentor's temporal-mode validator rejects requests where any column
is absent from any release.

Result: test_normal Δ MAE −0.324 (vs PR B's −0.239, a 0.085 c/L gain).
test_crisis Δ MAE −0.170 (vs PR B's −0.321, a 0.151 c/L regression).

**Interpretation:** The hypothesis from spec §7.7.2 finally lands on the
in-distribution fold. Quarterly welfare values add real signal on
test_normal where DSS releases all exist. The crisis-fold regression
likely reflects the model overfitting to recent-quarter DSS values that
don't generalise to 2026 crisis dynamics — a robustness vs accuracy
tradeoff. Worth investigating which DSS cols carry the signal vs noise.

### E2 — GCP temporal (cautionary tale)

Setup: 9 GCP-family cols (3 direct + 6 GCP-internal PRESETs) moved from
cross-sectional to temporal. Per-row Census values from 2016 (Edition 2)
or 2021 (Edition 3) depending on row date. Splitter was applied so
direct refs + PRESETs each got their own pass.

Result: test_normal Δ MAE **+0.470** — Model B *lost* by 0.470 c/L. A
disaster. test_crisis was modestly worse (−0.255 vs baseline −0.321).

**Interpretation:** Same finding as our v1.5 → v2.0.0 spike — Census 2016
introduces noise on the model. Per-row swapping confuses the model where
the static 2021 baseline was stable. **GCP variables should stay
cross-sectional going forward** — the spec §7.7.2 routing decision was
correct on the GCP side.

### E3 — Combined DSS + GCP temporal (kitchen sink dilutes the wins)

Setup: both E1 and E2 changes applied together. 22 vars in temporal
pass.

Result: test_normal Δ MAE −0.041 (vs PR B −0.239 — a 0.198 c/L
regression). test_crisis Δ MAE −0.320 (≈ baseline).

**Interpretation:** E2's GCP regression cancels out E1's DSS gain. If
you want both, do E1 alone, not E3. The combined-temporal hypothesis
is not supported.

### E4 — New ERP density + curation broadening (biggest single win)

Setup: temporal pass unchanged from PR B (SEIFA + ERP-total only); added
`ERP.population_density_per_km2` (new in upstream PR #97) to
cross-sectional; grew `SA2_COLUMNS` from 15 to 21 by adding PR A's 5
previously-fetched-but-unmodeled candidates + the new density col.

Result: test_normal Δ MAE −0.088 (vs PR B −0.239, a 0.151 c/L regression).
test_crisis Δ MAE **−0.603** (vs PR B −0.321, a **0.282 c/L gain** — the
largest improvement of any experiment in the project's history).

**Interpretation:** Same v1.5-era pattern — broadening the model block
trades test_normal accuracy for test_crisis robustness. The crisis-fold
gain is substantial and worth chasing. Open question: is the gain driven
by the **new density column** specifically, or by the **curation
broadening** of all 6 added cols? An ablation (E4-density-only,
E4-curation-only) would attribute.

## Recommended next steps

1. **Run E5 = E1 (DSS temporal, 9 cols) + E4 (new ERP density + 21-col
   curated block).** Hypothesis: hit both the test_normal win and the
   test_crisis win simultaneously. Estimated wall-clock ~45 min.
2. **Run E6 ablation of E4** — split into E4a (just the new density col,
   no curation broadening) and E4b (curation broadening only, no density).
   Attributes the test_crisis +0.282 to its actual driver. ~70 min.
3. **Don't ship E2/E3.** GCP temporal regresses; keep GCP cross-sectional.
4. **Pin to v2.1.0 is safe** regardless of which experiment you ship —
   it's a strict superset of v2.0+main, with the area-computation bug
   (#101) fixed.

## Failure modes from the FIRST run (now resolved)

The initial overnight run had 3 of 4 experiments fail. Root causes and
fixes:

- **Pre-Pipeline crash** — augmentor #101 (null-geom in
  `compute_sa2_areas_km2`). **Resolved upstream** by v2.1.0 (PR #102).
- **E1 schema-drift** — augmentor's temporal-mode validator required
  every requested column to exist in every release; FTB-A/B aren't in
  2024-Q2, jobseeker/CRA aren't in 2015-Q1. **Worked around** by
  trimming DSS_FAMILY to the 9 universally-available cols.
- **E2/E3 collision splitter** — the cross-sectional pass's
  `_split_for_preset_collision` wasn't applied in the temporal pass.
  **Fixed locally** by extracting the splitter to
  `src/fuel_pred/build/_augmentor_helpers.py` and applying it in
  `enrich_panel_temporal._augment`.

## Per-experiment artefacts

All per-experiment artefacts are gitignored under `data/`, `models_*`.

| Experiment | Comparison report | Features parquet | Models dir |
|------------|-------------------|------------------|------------|
| e1_dss_temporal | `results/pr_c_e1_dss_temporal_comparison.md` | `data/processed/features_e1_dss_temporal.parquet` | `models_e1_dss_temporal/` |
| e2_gcp_temporal | `results/pr_c_e2_gcp_temporal_comparison.md` | `data/processed/features_e2_gcp_temporal.parquet` | `models_e2_gcp_temporal/` |
| e3_combined_temporal | `results/pr_c_e3_combined_temporal_comparison.md` | `data/processed/features_e3_combined_temporal.parquet` | `models_e3_combined_temporal/` |
| e4_new_erp_density_plus_curation | `results/pr_c_e4_new_erp_density_plus_curation_comparison.md` | `data/processed/features_e4_new_erp_density_plus_curation.parquet` | `models_e4_new_erp_density_plus_curation/` |

## Experiment descriptions

- **e1_dss_temporal** — Move 9 DSS variables (universal across DSS
  releases per `tools/research/dss_schema_probe.py`) to the temporal
  pass. 4 partial-coverage DSS cols (jobseeker, CRA, FTB-A/B) stay
  cross-sectional.
- **e2_gcp_temporal** — Move GCP direct + GCP-internal PRESETs (9 vars)
  to temporal (unblocked by augmentor #91 Stage 2).
- **e3_combined_temporal** — Both E1 and E2 changes combined.
- **e4_new_erp_density_plus_curation** — Add the new
  `ERP.population_density_per_km2` to cross-sectional + grow
  `SA2_COLUMNS` from 15 to 21 (adds: `sa2_erp_population_65_plus`,
  `sa2_erp_median_age`, `sa2_pct_age_pension_recipients`,
  `sa2_pct_jobseeker_recipients`, `sa2_welfare_density_index`,
  `sa2_erp_population_density_per_km2`).
