# The Centrelink-day x SEIFA hypothesis: a useful null result

**Date:** 2026-05
**Status:** tested, partially-supported (main effect yes, interaction weak)
**TL;DR:** The NSW fuel cycle aligns with the fortnightly Centrelink payment calendar
and the model uses `cal_day_of_fortnight` as a clear main effect, but adding SEIFA
disadvantage as an interaction does *not* produce a visible per-quintile modulation
of that cycle. The augmentor still improves headline MAE (-0.391 c/L on test_normal)
— but via broad demographic conditioning of price *levels*, not via the hypothesised
fortnight *timing* modulation.

## The hypothesis

Centrelink pays most working-age welfare in Australia on a fortnightly cycle
(Wednesdays and Thursdays, anchored to a national pay calendar). The
`cal_day_of_fortnight` feature in `make_features.py` is anchored at 2016-07-04
(a Monday) to align with this cycle (spec §7.1).

The behavioural-economic hypothesis: fuel retailers in SA2s with high welfare
dependence (low SEIFA IRSD = more disadvantaged) might price-discriminate around
payment days, exploiting locally inelastic demand. If true, the NSW petrol cycle
shouldn't just have a fortnightly footprint — its *amplitude or phase* should
differ across the SEIFA gradient. That would have been a strong story for the
augmentor: demographic data unlocking a behavioural pattern invisible to
price-dynamics features alone. Spec §13 names this explicitly: SEIFA IRSD is
*"key for Centrelink-day interaction"* (spec.md line 165), and the EDA notebook
labels section 6 the *"augmentor-story chart"* that *"must be in the notebook"*
(`notebooks/01_eda.ipynb` cell 13).

## What we expected to see

Two empirical signatures would have constituted support:

1. **In `notebooks/01_eda.ipynb` §6**: the per-SEIFA-quintile residual lines
   (price minus 28-day rolling mean, grouped by `cal_day_of_fortnight`) should
   *diverge*. Specifically, low-SEIFA quintiles should sit higher around the
   post-payment window (days ~9-12, the post-Wednesday/Thursday peak of the
   cycle) than high-SEIFA quintiles, or show a sharper amplitude.

2. **In `notebooks/03_explainability.ipynb`**: the SHAP interaction scatter for
   `cal_day_of_fortnight x sa2_seifa_irsd_score` should show vertical colour
   stratification — at each day-of-fortnight, the SHAP values for low-SEIFA
   (blue) points should sit clearly above or below high-SEIFA (red) points.
   A clean interaction would look like two parallel envelopes coloured by
   quintile.

## What we actually saw

### The fortnight cycle is a real main effect

`results/shap/interaction_dof_seifa.png` shows the model has internalised the
fortnight cycle. The SHAP value for `cal_day_of_fortnight` is roughly flat and
near-zero on days 0-10 (range ~±0.04), then expands sharply on days 11-13 where
the per-row contribution spreads from -0.05 to +0.10. There's also a small
cluster of strongly negative SHAP values (-0.14 to -0.15) on days 0-2 — the
"early fortnight dip" the headline `results/README.md` flags. The model is using
day-of-fortnight; that part of the story holds.

### The SEIFA modulation is weak/absent

The same chart's colour channel — SEIFA IRSD score, blue (~870, more disadvantaged)
through purple to red (~1080, more advantaged) — does *not* stratify the SHAP
values. At every day-of-fortnight bin, blue, purple and red points are
interleaved through the full vertical spread of SHAP values. The late-fortnight
expansion on days 11-13 affects all colours roughly equally; the early-fortnight
dip on days 0-2 contains a mix of colours, and the most negative points
(SHAP ~ -0.15) actually appear pink-to-purple, not the blue end we'd expect if
disadvantaged SA2s were driving the dip. There is no day at which one SEIFA
band visibly sits above or below another.

In short: SEIFA does not pick out a different fortnight pattern from the rest
of the data.

### The augmentor still helped, just not via this mechanism

Headline `results/README.md`: Model B (with the 15-column SA2 block) beats
Model A by Δ MAE -0.391 c/L on test_normal (-6.2%), every SEIFA quintile
improves, and every brand benefits. That lift is real. But:

> *"No SA2 feature appears in the top 20 by gain importance, or the top 30 by
> mean |SHAP|, in either model."*

The augmentor's value is broad-and-shallow demographic conditioning of price
*levels* — a thin layer underneath the price-dynamics core that nudges many
predictions slightly — not a concentrated interaction unlock. SA2 features add
level signal, not timing signal.

## Why we think the hypothesis failed (or partially failed)

Plausible explanations, in rough order of credibility:

- **Other features absorb the variance first.** Retailers may well price-
  discriminate, but on signals the model already has — day of week, brand,
  competitor density, holiday flags. By the time SEIFA gets to "explain" the
  fortnight cycle, the residual it could attach to is small. The
  `results/README.md` caveat that SA2 columns are |r| > 0.5 collinear with the
  traffic/competitor blocks supports this read.
- **SEIFA IRSD is a station-static feature.** Every row for a given station
  carries the same IRSD score. The interaction is therefore identified entirely
  from *cross-station* variance in fortnight behaviour. If most of the
  fortnight-amplitude variance is *within* a station (driven by competitor
  dynamics on a given week), the cross-station SEIFA signal has little to bite
  on.
- **Cell-count thinness.** 14 fortnight days x 5 SEIFA quintiles = 70 cells.
  With the dominant variance coming from `lag_price_1` and Brent lags, the
  marginal interaction signal per cell may not survive the noise. LightGBM's
  default-depth trees can in principle find the interaction; that they don't
  rank it anywhere near the top suggests there's little of it to find.

## What we'd test if we had more time

- A **per-station-fixed-effects model** that removes cross-station level
  variance and lets the within-station fortnight cycle's modulation surface
  cleanly.
- A direct test on SA2s with **abnormally high Centrelink-recipient density**,
  using the `sa2_dss_*` columns that were added in the final 15-column block
  (carer-allowance, parenting-payment, youth-allowance recipients). These are
  more direct welfare-density measures than SEIFA IRSD, which is a composite.
- **Restrict to U91 in the bottom SEIFA quintile** and compare the fortnight
  cycle's amplitude (e.g. peak-to-trough cents/L) against the top quintile on
  the same date range. A direct effect-size estimate, not a SHAP interaction.

## Why this null result is worth keeping

The model improved with SA2 features. The mechanism is *demographic
conditioning of price levels*, not *demographic modulation of price timing*.
That distinction matters:

- For anyone designing an experiment on a different behavioural-economic
  hypothesis in transaction data, treat this as a prior: when your hypothesis
  is "interaction between time and a demographic stratum," expect the
  interaction to be hard to find even when the main effects are clear and
  even when the augmentor delivers a real headline lift.
- The augmentor's value is more interpretable as "broad calibration
  improvement" than "specific demographic interaction." That's a less
  exciting story than "we found price discrimination," but it's the honest
  one — and it's the one you can actually point at SHAP plots to defend.
- A null result on the *headline* hypothesis sharpens the interpretation of
  the lift that *is* there. The 0.391 c/L isn't coming from a hidden
  behavioural pattern; it's coming from many small demographic nudges. That
  reframes the augmentor's pitch from "unlocks new behavioural signal" to
  "improves calibration across the board," which is what the data supports.

## See also

- `notebooks/01_eda.ipynb` §6 — the per-quintile residual chart (the
  "augmentor-story chart")
- `notebooks/03_explainability.ipynb` §3 — the SHAP interaction plot
- `results/shap/interaction_dof_seifa.png` — the canonical interaction chart
  described above
- `results/README.md` caveat #1 — the headline version of this conclusion
- `spec.md` §13 — the original framing of the hypothesis
