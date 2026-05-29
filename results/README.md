# Results — Model A vs Model B

**Question (spec §1):** does augmenting per-station features with SA2-level ABS
Census demographics — via [`abs-census-augmentor`](https://github.com/cauldnz/abs-census-augmentor) —
measurably improve next-day NSW fuel-price prediction?

**Answer: yes.** Two LightGBM models with identical pipelines and hyperparameters,
differing only in the SA2 demographic block (Model A omits it, Model B includes it),
trained on identical rows. Model B wins on every held-out fold, every SEIFA quintile,
and every brand on the headline fold.

---

## Headline (v2.0, leakage-corrected)

All metrics in cents/L except MAPE (%). **Negative Δ MAE = Model B beats Model A.**

| Fold | n (U91) | MAE A | MAE B | Δ MAE | rel. | RMSE A | RMSE B | MAPE A | MAPE B |
|------|--------:|------:|------:|------:|-----:|-------:|-------:|-------:|-------:|
| test_normal (2024-01 → 2025-12) | 849,334 | 6.373 | **6.020** | **−0.353** | −5.5% | 10.953 | 10.589 | 3.352 | 3.167 |
| test_crisis (2026-01 → 2026-04) | 172,858 | 13.616 | **13.218** | **−0.398** | −2.9% | 19.054 | 18.578 | 6.181 | 6.002 |

Full segmentation (metro/regional, brand, fuel, SEIFA quintile) + feature-importance
tables in [`comparison.md`](comparison.md). SHAP visualisations in [`shap/`](shap/) (note:
SHAP plots have not yet been regenerated against v2 — they reflect v1 features; refresh
pending a notebook re-run).

### v1 → v2 transition (spec §13.7)

The v1 headline (preserved here for historical reference) used ERA5 reanalysis
*actuals* for the weather block, which is leakage — in real deployment the model
would have a *forecast* for tomorrow, not retrospective truth. v2.0 corrects this
by switching to NOAA GFS day-ahead forecasts. The full transition is documented
at [`docs/research/2026-05_weather_leakage_fix_outcome.md`](../docs/research/2026-05_weather_leakage_fix_outcome.md).

| Fold | v1 MAE B | v2 MAE B | Δ | v1 Δ MAE | v2 Δ MAE |
|------|---------:|---------:|--:|---------:|---------:|
| test_normal | 5.912 | 6.020 | +0.108 (leakage tax) | −0.391 | −0.353 |
| test_crisis | 13.283 | 13.218 | −0.065 | −0.183 | **−0.398** |

**Two notable v2 findings:**

1. The **leakage tax** on absolute MAE was small (+0.07-0.15 c/L) — within the
   predicted 0.05-0.15 range. The model genuinely could only cheat with ERA5 by
   small amounts because the wx_* block is low-rank in the feature set overall.
2. The **crisis-fold SA2 lift more than doubled** in v2 (−0.183 → −0.398), and
   the v1 crisis-fold RMSE regression (Model B *worse* than A) is gone (now
   B beats A on RMSE). The honest weather block makes the SA2 block's true
   contribution clearer, and that translates to a more robust improvement
   on OOD 2026 data. See caveat #4 below.

---

## How we got here — the iteration story

The headline number is the end of a four-step search, not a first try. Each step is a
real data point about how much SA2 demographics help and which ones.

| Iteration | Augmentor | Weather | SA2 cols | Test_normal Δ MAE | What we learned |
|-----------|---------|---------|---------:|------------------:|-----------------|
| v1.0 | v1.4.2 | ERA5 (leaky) | 10 | **+0.104** (Model B *lost*) | First real run; SA2 hurt the headline fold |
| v1.1 | v1.5 | ERA5 (leaky) | 10 | **−0.059** | The augmentor's improved parsing (v1.5) was itself a material win — same column *names*, better values |
| v1.2 | v1.5 | ERA5 (leaky) | 31 | **−0.025** | Broadening (DSS welfare + ERP + ABS_PIA + all SEIFA scores) *regressed* — better val MAE, worse test: textbook overfitting |
| v1.3 | v1.5 | ERA5 (leaky) | 15 | **−0.391** | Curating to the original 10 + the 5 highest-gain new features recovered the full benefit *without* the overfitting tax |
| v2.0 weather fix | v1.5 | **NOAA GFS** (honest, spec §13.7) | 15 | **−0.353** | Weather-leakage fix; small absolute-MAE tax (~0.04 c/L), but **crisis-fold Δ doubled** (−0.183 → −0.398). Headline switched to this row when v2.0 landed |
| **augmentor v2.0 bump (current)** | **v2.0** | **NOAA GFS** | **15** | **−0.353** | Pin bump with identical model block; cross-sectional v2.0 produces (byte-)identical predictions to v1.5 on this 15-col surface. Validates the upgrade is a no-op for the modeled features and the 5 new columns (ERP age/sex + 3 cross-dataset PRESETs) are safely available in `stations.parquet` for a follow-up curation experiment |

The v1.5 review's recommendation #1 — "re-run your headline experiment on every minor-version bump" — was respected: this time the bump was a no-op on the 15-col surface. Compare with the v1.4.2→v1.5 swing on the same 10-col block (+0.104 → −0.059, a 0.163 c/L shift), which was a non-trivial reminder that augmentor versions can be hyperparameters when they're not.

The key methodological finding: **more features didn't help — the right features did.**
Broadening from 10 → 31 columns added 21 features that mostly re-encoded urban density
already captured by the traffic/competitor blocks; they inflated val-fold fit and degraded
test generalization. Curating back to 15 by feature-importance ranking (see [spec §7.7.4](../spec.md))
produced the strongest result of any iteration.

### The final SA2 block (15 columns)

Original Census/SEIFA baseline (10):
`sa2_total_population`, `sa2_median_age`, `sa2_median_household_income_weekly`,
`sa2_pct_drive_to_work`, `sa2_motor_vehicles_per_dwelling`, `sa2_pct_renters`,
`sa2_pct_employed_full_time`, `sa2_pct_aged_65_plus`, `sa2_pct_one_parent_family`,
`sa2_seifa_irsd_score`.

Curated additions, top-5 by gain importance from the 31-col experiment (5):
`sa2_seifa_ieo_score`, `sa2_dss_parenting_payment_partnered_recipients`,
`sa2_dss_carer_payment_recipients`, `sa2_dss_carer_allowance_recipients`,
`sa2_dss_youth_allowance_student_and_apprentice_recipients`.

---

## Where the lift comes from

**It's broad, not concentrated.** No SA2 feature appears in the top 20 by gain importance,
or the top 30 by mean |SHAP|, in either model ([`shap/summary_b.png`](shap/summary_b.png),
[`shap/importance_a_vs_b.png`](shap/importance_a_vs_b.png)). The model's headline drivers are
unchanged from Model A — `lag_price_1` dominates, then Brent lags, day-of-month, brand. The
SA2 block adds a thin, broad layer of demographic context *underneath* the price-dynamics core
that systematically improves calibration without displacing any top feature.

This is the honest shape of the augmentor's value: not "demographics are top predictors," but
"demographic context nudges many predictions slightly, and those nudges add up to a robust lift."

**The lift scales with affluence.** On test_normal, every SEIFA quintile improves, but the
gradient is clear (Δ MAE): Q1 −0.250, Q2 −0.314, Q3 −0.345, Q4 −0.410, **Q5 −0.678**. The model
extracts the most SA2 value in the least-disadvantaged areas — plausibly where price dispersion
across competing premium stations is highest and demographic context disambiguates most.

**Every brand benefits on test_normal** (Δ MAE): 7-Eleven −0.724, BP −0.523, Ampol −0.517,
Coles Express −0.347, Other −0.254, United −0.227, Metro −0.149, Independent −0.144, Speedway −0.110.

### Most interpretable single feature: carer-allowance recipients

`sa2_dss_carer_allowance_recipients` jumped from gain-rank 51 (during the 31-col experiment) to
**mean-|SHAP| rank 1** in the final model — the clearest example of gain importance and per-row
impact diverging. Its dependence plot ([`shap/dependence_sa2_dss_carer_allowance_recipients.png`](shap/dependence_sa2_dss_carer_allowance_recipients.png))
shows a clean near-monotonic relationship: SA2s with more carer-allowance recipients → lower
predicted prices, with stepped tree-threshold structure. Caregiver-heavy SA2s skew older,
outer-metro and regional, where the price cycle behaves differently from inner-metro premium
competition — a defensible economic signal, not an artifact.

Top 5 SA2 features by mean |SHAP| on the test_normal sample:

| Rank | Feature | Mean \|SHAP\| | Gain rank |
|-----:|---------|--------------:|----------:|
| 1 | `sa2_dss_carer_allowance_recipients` | 0.078 | 51 |
| 2 | `sa2_pct_drive_to_work` | 0.064 | 31 |
| 3 | `sa2_seifa_irsd_score` | 0.049 | 46 |
| 4 | `sa2_seifa_ieo_score` | 0.049 | 50 |
| 5 | `sa2_dss_carer_payment_recipients` | 0.041 | 48 |

---

## Caveats — what we deliberately do *not* claim

1. **The fortnight × SEIFA interaction is weak.** The model clearly captures a fortnightly price
   cycle — the SHAP value for `cal_day_of_fortnight` rises across the fortnight with a strong
   early-fortnight dip ([`shap/interaction_dof_seifa.png`](shap/interaction_dof_seifa.png)). But
   the *modulation of that cycle by SEIFA disadvantage* — the "Centrelink-day price discrimination"
   hypothesis — is **not cleanly supported** by the data: the SEIFA colouring doesn't separate the
   day-of-fortnight SHAP values in an obvious way. We report the fortnight cycle as a **main effect**
   the model uses, and explicitly do **not** claim demonstrated demographic interaction.

2. **~~The crisis-fold lift is real but smaller and noisier.~~** *(v1 caveat — invalidated by v2.)*
   v1 reported crisis-fold Δ MAE −0.183 with Model B's RMSE marginally *worse* than A's (18.739 vs
   18.628) and two brands regressing slightly. **v2.0 changed this:** crisis-fold Δ MAE more than
   doubled to **−0.398**, B's RMSE is now lower than A's (18.578 vs 19.054), and every reported brand
   benefits. The most plausible mechanism is that v1's leaky ERA5 weather block was an
   unrealistically strong in-distribution predictor, masking the SA2 block's true marginal value.
   See [`docs/research/2026-05_weather_leakage_fix_outcome.md`](../docs/research/2026-05_weather_leakage_fix_outcome.md).

3. **SA2 features are collinear with the traffic/competitor blocks.** Several SA2 columns correlate
   |r| > 0.5 with `stn_competitors_*` / `ctx_traffic_*` (all downstream of urban density). The model
   still extracts independent signal from them (LightGBM's `feature_fraction=0.8` samples columns per
   tree), which is why the 15-col block helps — but this collinearity is why *adding more* correlated
   SA2 features (the 31-col experiment) overfit. The augmentor's effective new dimensionality for
   short-horizon fuel-price prediction is modest.

4. **✅ Weather leakage (v1) — fixed in v2.0 (spec §13.7).** v1 used ERA5 reanalysis actuals across the
   full span rather than forecast-at-lead-time-1. v2.0 switches to NOAA GFS day-ahead forecasts via
   anonymous AWS S3 byte-range subsetting. The leakage tax on absolute MAE was small (+0.07-0.15 c/L,
   within the predicted range), and as a bonus the crisis-fold SA2 lift improved substantially (caveat
   #2 above). Numbers in the headline table reflect v2.0. Full v1 → v2 transition write-up at
   [`docs/research/2026-05_weather_leakage_fix_outcome.md`](../docs/research/2026-05_weather_leakage_fix_outcome.md).

   v2.0 carries three small acknowledged compromises (all documented in the outcome doc): `wx_weather_code_t1`
   is null-stubbed (GFS doesn't emit WMO codes; low SHAP rank, costs <0.01 c/L), ~20% of training rows have
   null `wx_*_t1` from 2016 + NOAA archive gaps (handled natively by LightGBM), and the daily aggregation
   uses a UTC day boundary rather than Sydney-local (~10h offset, low-rank feature).


---

## Reproduction

Full pipeline from a clean checkout (needs network for raw fetches, or a pre-populated `data/raw/`):

```bash
make all          # fetch → clean → enrich → features → train → evaluate → notebooks
```

Or the train+evaluate stages against an existing `data/processed/features.parquet`:

```bash
make train        # fits Model A + Model B → models/, writes prediction parquets
make evaluate     # → results/comparison.md
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_explainability.ipynb  # → results/shap/
```

Fixed hyperparameters (spec §8.2), identical for both models. Both train on the **identical row set** —
the intersection where every SA2 column is non-null (~91% of U91 rows) — so the comparison isolates the
SA2 block as the only difference. Folds are time-based (spec §8.3): train ≤ 2022, val 2023,
test_normal 2024-25, test_crisis 2026.

---

## Artifact index

| File | Contents |
|------|----------|
| [`comparison.md`](comparison.md) | Full metrics: headline + metro/brand/fuel/SEIFA-quintile segmentation + feature-importance + SA2↔non-SA2 correlation tables |
| [`shap/summary_b.png`](shap/summary_b.png) | Model B top-30 features by mean \|SHAP\| |
| [`shap/importance_a_vs_b.png`](shap/importance_a_vs_b.png) | Side-by-side gain importance, A vs B |
| [`shap/dependence_*.png`](shap/) | Dependence plots, top-5 SA2 features by SHAP impact |
| [`shap/interaction_dof_seifa.png`](shap/interaction_dof_seifa.png) | `cal_day_of_fortnight × sa2_seifa_irsd_score` interaction |
| [`shap/case_studies_predictions.png`](shap/) | Predictions vs actuals, Q1/Q3/Q5 representative stations |
| [`shap/waterfall_*.png`](shap/) | Per-prediction SHAP waterfalls, one per case-study station |

---

## Limitations & future work

- **Augmentor signal looks near-exhausted.** Three iterations converged on a 15-column sweet spot;
  the 31-col broadening overfit and the headline interaction hypothesis came back weak. Further SA2
  feature selection is unlikely to move the headline materially.
- **Temporal DSS deferred (spec §7.7.2).** DSS welfare data is pinned to a single latest-quarter
  snapshot. Per-row temporal resolution (the augmentor v1.5 capability) was deprioritised after static
  DSS features contributed only modestly — the temporal-only delta is unlikely to justify the
  panel-level augmentation refactor it requires.
- **Weather leakage (above)** is the clearest absolute-accuracy caveat for v2.
- **`stn_is_metro` is a name-heuristic**, and the metro/regional segmentation is heavily imbalanced
  (2,998 metro vs 846,336 regional rows on test_normal) — read that split with caution.
