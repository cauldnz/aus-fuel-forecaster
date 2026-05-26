# SA2 feature curation: 10 → 31 → 15 columns

**Date:** 2026-05
**Status:** completed; final 15-col block shipped in v1
**TL;DR:** When adding an augmentor block to a predictive model, broadening the
column set hurt generalisation; trimming back to the original baseline plus the
five highest-gain new features (15 columns total) produced the strongest
result of any iteration. More features did not help — the right features did.

## The question

The SA2 demographic block is the *only* difference between Model A and Model B
(spec §8.4). Picking which augmentor variables to expose looks trivial —
request them, train, see what happens — but the choice is not obvious:

1. **Naturally-correlated demographics.** Census, SEIFA, ERP, ABS Personal
   Income and DSS welfare counts describe the same SA2 from overlapping
   angles. Most of them co-vary with urban density.
2. **Collinearity with existing blocks.** The pipeline already carries
   `stn_competitors_within_*` and `ctx_traffic_*` features that proxy urban
   density. Several SA2 candidates correlate `|r| > 0.5` with those — e.g.
   `sa2_pct_drive_to_work` ↔ `ctx_traffic_5km_radius_count` at `-0.760`
   (`results/comparison.md`, "High correlations" table).
3. **The augmentor surface is wider than the model needs.** The v1.5
   augmentor exposes 21 columns beyond the original Census/SEIFA-IRSD
   baseline. No a-priori reason to prefer "all 31" over "the original 10".

The methodology question: given a noisy A/B signal worth roughly 0.4 cents/L
of MAE, how do we pick the SA2 columns without (a) over-fitting the val fold
or (b) discarding columns that contribute small but real lift?

## The four iterations

Setup identical across runs (spec §8): LightGBM with frozen `LGBM_PARAMS`,
time-based train/val/test_normal/test_crisis split, both models restricted
to the row intersection where every SA2 column is non-null. Only the SA2
column tuple changed.

| # | Config | Cols | Test_normal Δ MAE | Lesson |
|--:|--------|-----:|------------------:|--------|
| 1 | v1.4.2 augmentor, 10-col baseline | 10 | **+0.104** (Model B *lost*) | Upstream PRESET parser was wrong; same column names, broken values. Not a feature-selection failure. |
| 2 | v1.5 augmentor, 10-col baseline | 10 | **−0.059** | Fixing the parser flipped the same column list from loss to win. |
| 3 | v1.5 augmentor, broadened to 31 | 31 | **−0.025** | 21 added DSS / ERP / ABS_PIA / SEIFA columns *regressed* the headline. Val improved (4.78 vs 4.85; spec §7.7.4), test got worse. Textbook overfitting. |
| 4 | v1.5 augmentor, curated to 15 (final) | 15 | **−0.391** | Original 10 + top 5 by gain from the 31-col run recovered the full benefit without the overfitting tax. |

Iteration 1's negative was an upstream parser bug — v1.4.2 PRESETs
referenced GCP columns that did not exist (synthetic fixtures encoded the
same broken names; tests passed). Lesson: validate the *values* an
augmentor returns, not just that variable refs parse (spec §7.7.3).
Iteration 3 is the interesting one and the reason this note exists.

## Methodology — how we got from 31 to 15

The procedure after iteration 3 regressed:

1. **Train the 31-col model to completion** and dump LightGBM's
   `feature_importance(importance_type="gain")` for every feature.
2. **Express gain as a percentage of total gain** so the threshold is
   comparable across runs with different tree counts.
3. **Set a floor at 0.02% of total gain.** The bottom 16 of the 21 new
   features all came in at ≤ 0.01% gain (`results/comparison.md`, "Where
   SA2 features rank in Model B"). 0.02% was a natural shoulder in the
   rank distribution, not a principled number — partly empirical.
4. **Keep the original 10 unconditionally.** Validated across three prior
   iterations; iteration 3 was about broadening, not displacement.
5. **Add only new features ≥ 0.02% gain.** Five qualified: one extra SEIFA
   score (IEO) and four DSS welfare counts.

**Why gain rather than SHAP for the cut?** Gain is what the model actually
used at training — it tells you which features absorbed splits and reduced
loss. SHAP measures per-row attribution at evaluation, a different quantity;
useful for explaining a deployed model, less so for a coarse keep/drop
decision. Gain is also cheap (falls out of training); SHAP requires a
separate explainer pass.

We cross-referenced gain rank against mean |SHAP| once the curated model
was trained, and the divergence is striking. `sa2_dss_carer_allowance_recipients`
sat at **gain-rank 51** in the 31-col run (just above the floor — almost
dropped) but ended up at **mean-|SHAP| rank 1 among all SA2 features** in
the final 15-col model (`results/README.md`). A feature that barely cleared
a gain cut can carry the per-row interpretability story.

## The 16 features we dropped from the 31-col block

All ranked below the 0.02%-gain floor (≤ 0.01% gain) in the 31-col run. The
`results/comparison.md` SA2 importance table reports the top 15 (the final
block); the 16 below all sat lower.

| Dropped column | Reason (all also < 0.02% gain) |
|----------------|--------|
| `sa2_seifa_irsad_score` | Near-duplicate of IRSD (two-direction vs disadvantage-only) |
| `sa2_seifa_ier_score` | Economic-resources continuum captured by household income + IRSD |
| `sa2_erp_population_total` | Redundant with `sa2_total_population` for short-horizon prediction |
| `sa2_pia_median_total_income` | Redundant with `sa2_median_household_income_weekly` |
| `sa2_pia_mean_total_income` | Ditto |
| `sa2_pia_income_earners_count` | Correlates with population + employment columns |
| `sa2_pia_median_age_of_earners` | `|r|=-0.555` with `stn_competitors_within_5km` |
| `sa2_dss_age_pension_recipients` | Redundant with `sa2_median_age` + `sa2_pct_aged_65_plus` |
| `sa2_dss_jobseeker_payment_recipients` | Disadvantage signal redundant with SEIFA IRSD |
| `sa2_dss_disability_support_pension_recipients` | — |
| `sa2_dss_parenting_payment_single_recipients` | Partnered variant ranked higher and was kept |
| `sa2_dss_youth_allowance_other_recipients` | Student+apprentice variant ranked higher and was kept |
| `sa2_dss_commonwealth_rent_assistance_recipients` | Rental signal redundant with `sa2_pct_renters` |
| `sa2_dss_commonwealth_seniors_health_card_recipients` | Older-population signal redundant |
| `sa2_dss_family_tax_benefit_a_recipients` | Family signal redundant with `sa2_pct_one_parent_family` |
| `sa2_dss_family_tax_benefit_b_recipients` | Ditto |

Most drops are redundant rather than useless: each dropped column carries
some signal in isolation, but the marginal information after the 15
retained features is too small to extract reliably against the noise the
column also brings.

## The 5 features we added (and kept)

From `src/fuel_pred/train/feature_blocks.py`:

| Added column | Gain rank (31-col) | Mean-|SHAP| rank (SA2 only) | Rationale |
|--------------|-------------------:|----------------------------:|-----------|
| `sa2_seifa_ieo_score` | 51 (~0.02%) | 4 | Only Education + Occupation SEIFA; distinct from IRSD's disadvantage-only continuum. |
| `sa2_dss_parenting_payment_partnered_recipients` | 45 (~0.04%) | not in top 5 | Highest-impact new feature by gain. |
| `sa2_dss_carer_payment_recipients` | 48 (~0.02%) | 5 | Care-giver demographic proxy. |
| `sa2_dss_carer_allowance_recipients` | 50 (~0.02%) | **1** | Broader complement to carer_payment. Gain barely cleared the floor; SHAP put it at the top — the divergence case. |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 49 (~0.02%) | not in top 5 | Young-cohort proxy. Kept despite `|r|=+0.656` with `stn_competitors_within_5km` — generational signal is distinct. |

`sa2_pct_drive_to_work` (in the original 10) ranks mean-|SHAP| #2 in the
final model despite `|r|=-0.760` with `ctx_traffic_5km_radius_count`.
Collinearity does not make a feature dead weight if it carries enough
independent signal to clear the gain bar.

## What generalises to other prediction problems

- **More features ≠ better.** A broader column set inflates val-fold fit
  while degrading test generalisation. Trust the held-out fold.
- **Gain ≥ ~0.02% of total is a defensible floor for a tree-based model.**
  Empirical — verify the rank distribution has a shoulder there before
  reusing the number.
- **Gain rank can diverge sharply from per-row impact.** Use gain for the
  *cut* (cheap, decision-relevant); use SHAP for the *narrative* (expensive,
  human-relevant). A feature barely above the gain floor can be the top
  per-row contributor.
- **Collinearity is the real enemy in demographic data.** Catalogue
  correlations against existing density proxies before broadening; if you
  include a `|r| > 0.5` feature, verify it brings independent signal.
- **Drop the redundant, not the absolutely-noisy.** Most dropped columns
  carried some signal in isolation — they were out-competed by a retained
  column with the same signal more cleanly. The cut is about marginal, not
  absolute, information.
- **Validate the augmentor's values, not just that variable refs parse.**
  Iteration 1 was an upstream parser bug; a 5-second probe-fetch would
  have caught it.

## What we would do differently

The 31-column broadening was worth doing — without it we had no defensible
basis to pick between "10 cols" and any other combination. We would do it
again, but treat the 31-col run up front as a rank-elicitation step, not a
candidate production model.

The 0.02%-gain threshold is partly arbitrary. A stricter cut (e.g. 0.05%)
would have left us with the original 10 plus only
`sa2_dss_parenting_payment_partnered_recipients`; we did not re-train at
that cut to verify whether 11 columns would beat 15. We know 15 beats 10
and 31; we do not know it is optimal. We stopped because the marginal
value of further search was low against the lift we already had.

Left on the table: per-quarter temporal DSS (spec §7.7.2 — top static
signal is ~0.04% gain; the temporal hypothesis would have to live entirely
in quarter-to-quarter variation), and a finer threshold sweep at
0.01% / 0.05% / 0.10% to map the signal cliff.

## See also

- `results/README.md` — headline result, SHAP analysis, iteration table
- `results/comparison.md` — full per-segment metrics, SA2 gain ranking,
  SA2 ↔ non-SA2 correlation table
- `spec.md` §7.7 / §7.7.4 — SA2 block spec, curation reasoning, kept/dropped
  table, iteration metrics with val MAE
- `src/fuel_pred/train/feature_blocks.py` — canonical `SA2_COLUMNS` tuple
- `src/fuel_pred/config.py` — `AUGMENTOR_VARIABLES`, which still requests
  all 31 columns so they remain available for ablation studies even though
  the model only consumes the 15 in `SA2_COLUMNS`
