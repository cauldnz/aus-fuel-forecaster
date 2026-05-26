# Research notes

This directory contains research spikes and retrospectives for `aus-fuel-forecaster`. Both kinds get committed:

- **Forward-looking spikes** — scoping a feature before building it. Sources surveyed, APIs probed, feature design proposed, decision gates defined. Linked from the relevant `spec.md` §13 backlog item.
- **Retrospectives and null results** — what we tried, what worked, what didn't. Committed even when the answer was "don't build it" — the next person on a similar problem benefits from seeing the dead-end.

The convention exists because research artefacts often have lasting value beyond the immediate problem. Someone forking this repo to attack a different prediction problem (different geography, different target, different demographic dataset) gets to see what we tried and why, without re-running the spike.

## Naming

`YYYY-MM_short_topic_slug.md` — date sorts the list chronologically; the slug names the topic.

## Index

### Forward-looking (waiting to be built)

| File | Topic | Spec ref |
|---|---|---|
| [`2026-05_major_events_features.md`](2026-05_major_events_features.md) | Spatial event features (AFL/NRL fixtures, major venues, long weekends). Gated on a Phase 0 EDA check. **EDA gate run — see [`2026-05_major_events_eda_outcome.md`](2026-05_major_events_eda_outcome.md) (verdict: PROCEED with caveats). Phase 1 sanity check run — see [`2026-05_major_events_phase1_outcome.md`](2026-05_major_events_phase1_outcome.md) (verdict: STOP, venue features regress test MAE).** | spec §13.6 |
| [`2026-05_weather_leakage_fix.md`](2026-05_weather_leakage_fix.md) | Switch Open-Meteo Archive → Historical Forecast API + 1-day join shift. ~2.5 sessions of work. | spec §13.7 |

### Retrospectives (work already done)

| File | Topic |
|---|---|
| [`2026-05_sa2_feature_curation.md`](2026-05_sa2_feature_curation.md) | The 10 → 31 → 15 column iteration: methodology for curating an augmentor block. |
| [`2026-05_abs_census_augmentor_v1.5_review.md`](2026-05_abs_census_augmentor_v1.5_review.md) | What changed v1.4.2 → v1.5 and why the parsing fix alone moved Δ MAE 0.16 c/L on the same 10-column schema. |
| [`2026-05_centrelink_seifa_interaction.md`](2026-05_centrelink_seifa_interaction.md) | Honest null result: the fortnight cycle is a strong *main effect*, but the SEIFA *interaction* is weak. The augmentor adds level signal, not timing signal. |

## See also

- `docs/troubleshooting/` — operational tips (Podman networking, Python environment gotchas) that don't belong here
- `results/README.md` — the experimental write-up for the v1 headline result
- `spec.md` §13 — the open-questions / backlog section that links to forward-looking spikes
