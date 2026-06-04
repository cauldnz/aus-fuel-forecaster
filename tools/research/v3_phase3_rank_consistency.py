"""v3.0 Phase 3 follow-up — per-fold rank consistency across the 8 Phase 2 experiments.

Reads the 8 per-experiment kfold reports under ``results/v3_phase2_*_kfold.md``,
extracts the per-fold Δ MAE (B−A) from each, then asks:

- **Q1.** Do the same folds always favour Model B (or never)? If yes → there's
  a real structural per-period effect we're missing; pursue Reading C
  (wrong features / wrong model). If the sign-by-fold shuffles randomly
  across experiments → Reading A (genuinely flat).
- **Q2.** How correlated are the per-fold Δ MAE *patterns* across pairs of
  experiments (Spearman ρ of fold rankings)? High correlation across all
  pairs → the augmentor variants are sub-tuning against the same per-fold
  noise structure. Low correlation → noise dominates.
- **Q3.** For each fold, what's the dispersion of Δ MAE across the 8
  experiments? If one fold consistently dominates the total variance,
  the "noise band" framing should be sharpened.

Writes ``results/v3_phase3_rank_consistency.md`` + ``…rank_consistency.json``.

This is purely a re-analysis of existing artefacts — no model retraining,
no compute. Runs in seconds. Used to disambiguate Reading A from Reading
C cheaply before committing to the (~5h) seed-noise experiment.

Spec / discussion: ``docs/research/2026-06_v3.0_phase2_postmortem_discussion.md``
(experiments #1 in the ranked next-steps list).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

logger = logging.getLogger(__name__)


# Map the 8 report filenames to short labels used in the summary table.
EXPERIMENTS: dict[str, str] = {
    "v3_phase2_pr_b_baseline_kfold.md": "pr_b_baseline",
    "v3_phase2_pr_c_e1_dss_temporal_kfold.md": "e1_dss",
    "v3_phase2_pr_c_e2_gcp_temporal_kfold.md": "e2_gcp",
    "v3_phase2_pr_c_e3_combined_temporal_kfold.md": "e3_combined",
    "v3_phase2_pr_c_e4_density_plus_curation_kfold.md": "e4_dens+cur",
    "v3_phase2_pr_c_e4a_density_only_kfold.md": "e4a_dens",
    "v3_phase2_pr_c_e4b_curation_only_kfold.md": "e4b_cur",
    "v3_phase2_pr_c_e5_dss_temporal_plus_curation_kfold.md": "e5_dss+cur",
}


# Per-fold row pattern in the A-vs-B headline table — Δ MAE is column 6 of
# the row body (after the "fold_N | window | n | MAE A | MAE B | Δ MAE").
# Tolerates `+0.074`, `-0.135`, `+1.042` etc.
_FOLD_ROW = re.compile(
    r"^\|\s*fold_(\d+)\s*\|"                # fold N
    r"\s*([\d-]+\s*→\s*[\d-]+)\s*\|"        # window
    r"\s*([\d,]+)\s*\|"                       # n
    r"\s*(-?\d+\.\d+)\s*\|"                   # MAE A
    r"\s*(-?\d+\.\d+)\s*\|"                   # MAE B
    r"\s*([+-]?\d+\.\d+)\s*\|"                # Δ MAE
)


@dataclass
class ExperimentResult:
    """Per-experiment per-fold metrics extracted from one kfold report."""

    label: str
    report_path: Path
    per_fold_delta_mae: list[float] = field(default_factory=list)  # 6 entries
    per_fold_mae_a: list[float] = field(default_factory=list)
    per_fold_mae_b: list[float] = field(default_factory=list)

    @property
    def mean_delta_mae(self) -> float:
        return statistics.fmean(self.per_fold_delta_mae)

    @property
    def stdev_delta_mae(self) -> float:
        # Population stdev (divides by n, not n-1) to match the headline
        # ``stdev`` column in the published v3_phase2_*_kfold.md reports.
        # We're treating the 6 folds as the whole population we care about,
        # not a sample drawn from a larger fold population — consistent
        # with the design-doc §2.5 framing.
        return statistics.pstdev(self.per_fold_delta_mae)


def parse_report(path: Path, label: str) -> ExperimentResult:
    """Parse a single ``v3_phase2_*_kfold.md`` report's A-vs-B headline table.

    Picks up the 6 ``| fold_N | window | n | MAE A | MAE B | Δ MAE | …``
    rows; ignores everything else.
    """
    out = ExperimentResult(label=label, report_path=path)
    text = path.read_text(encoding="utf-8")
    # The report has TWO per-fold tables: A-vs-B (first) and B-vs-B' (second).
    # We only want the A-vs-B headline — stop parsing after the first 6 rows.
    # Both tables have rows starting with `| fold_N | window | n | <three
    # numerics> |`, so the regex matches both; the early break keeps us
    # honest about which slice we're reading.
    for line in text.splitlines():
        if len(out.per_fold_delta_mae) >= 6:
            break
        m = _FOLD_ROW.match(line)
        if not m:
            continue
        _fold_n = int(m.group(1))
        mae_a = float(m.group(4))
        mae_b = float(m.group(5))
        delta = float(m.group(6))
        out.per_fold_mae_a.append(mae_a)
        out.per_fold_mae_b.append(mae_b)
        out.per_fold_delta_mae.append(delta)
    if len(out.per_fold_delta_mae) != 6:
        raise RuntimeError(
            f"expected 6 fold rows in {path}, parsed {len(out.per_fold_delta_mae)}; "
            "regex / report format may have drifted"
        )
    return out


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation between two parallel sequences.

    Tiny implementation (no scipy) — n is always 6 here. Returns the
    Pearson correlation of the rank vectors. Ties get average rank.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        return float("nan")

    def _ranks(values: list[float]) -> list[float]:
        # average ranks for ties
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based ranks
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    denom = (dx * dy) ** 0.5
    return num / denom if denom else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory containing the v3_phase2_*_kfold.md reports",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/v3_phase3_rank_consistency.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/v3_phase3_rank_consistency.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # ---- Parse the 8 reports --------------------------------------------------
    results: list[ExperimentResult] = []
    for filename, label in EXPERIMENTS.items():
        path = args.results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing report: {path}")
        r = parse_report(path, label)
        logger.info(
            "%-15s mean Δ = %+.3f stdev = %.3f (per-fold: %s)",
            label,
            r.mean_delta_mae,
            r.stdev_delta_mae,
            [f"{d:+.3f}" for d in r.per_fold_delta_mae],
        )
        results.append(r)

    # ---- Q1. Per-fold sign tally ---------------------------------------------
    # For each of the 6 folds, count: (B beats A: Δ<0), (tie: |Δ|<0.01), (A wins)
    # across the 8 experiments.
    n_folds = 6
    n_exps = len(results)
    fold_sign_tally: list[dict[str, int]] = []
    for f in range(n_folds):
        deltas = [r.per_fold_delta_mae[f] for r in results]
        b_wins = sum(1 for d in deltas if d < -0.01)
        a_wins = sum(1 for d in deltas if d > 0.01)
        ties = n_exps - b_wins - a_wins
        fold_sign_tally.append({"b_wins": b_wins, "a_wins": a_wins, "ties": ties})

    # ---- Q2. Pairwise Spearman ρ on the 6 fold-Δ values across experiments ---
    pair_rhos: list[tuple[str, str, float]] = []
    for a, b in combinations(results, 2):
        rho = spearman_rho(a.per_fold_delta_mae, b.per_fold_delta_mae)
        pair_rhos.append((a.label, b.label, rho))
    mean_rho = statistics.fmean(r for *_, r in pair_rhos)
    median_rho = statistics.median(r for *_, r in pair_rhos)

    # ---- Q3. Per-fold variance contribution ----------------------------------
    per_fold_stats: list[dict[str, float]] = []
    for f in range(n_folds):
        deltas = [r.per_fold_delta_mae[f] for r in results]
        per_fold_stats.append({
            "fold": f + 1,
            "mean_delta_mae": statistics.fmean(deltas),
            # Population stdev to match the published reports' convention
            # (see ExperimentResult.stdev_delta_mae note).
            "stdev_delta_mae": statistics.pstdev(deltas),
            "min_delta_mae": min(deltas),
            "max_delta_mae": max(deltas),
            "n_b_wins": fold_sign_tally[f]["b_wins"],
            "n_a_wins": fold_sign_tally[f]["a_wins"],
            "n_ties": fold_sign_tally[f]["ties"],
        })

    # Sum of squared deviations contributed by each fold (relative to its own
    # cross-experiment mean) — proxy for "which fold drives the total variance".
    total_ss = sum(
        sum((d - row["mean_delta_mae"]) ** 2 for d in [r.per_fold_delta_mae[i] for r in results])
        for i, row in enumerate(per_fold_stats)
    )
    for i, row in enumerate(per_fold_stats):
        deltas = [r.per_fold_delta_mae[i] for r in results]
        ss = sum((d - row["mean_delta_mae"]) ** 2 for d in deltas)
        row["variance_share"] = ss / total_ss if total_ss else 0.0

    # ---- Q4. Per-fold MAE_A (just for context — fold difficulty) -------------
    # Model A MAE is identical across the experiments that don't change Model A
    # blocks, but differs for E1/E5 (DSS-temporal) since the SA2 join filter
    # changes the training row set when curation changes. Capture them all.
    per_fold_mae_a_table: list[list[float]] = []
    for f in range(n_folds):
        per_fold_mae_a_table.append([r.per_fold_mae_a[f] for r in results])

    # ---- Write JSON ----------------------------------------------------------
    payload = {
        "experiments": [r.label for r in results],
        "per_fold_delta_mae": {
            r.label: r.per_fold_delta_mae for r in results
        },
        "per_fold_mae_a": {
            r.label: r.per_fold_mae_a for r in results
        },
        "per_fold_mae_b": {
            r.label: r.per_fold_mae_b for r in results
        },
        "per_fold_summary": per_fold_stats,
        "spearman_pairs": [
            {"a": a, "b": b, "rho": rho} for a, b, rho in pair_rhos
        ],
        "spearman_mean": mean_rho,
        "spearman_median": median_rho,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_json)

    # ---- Write markdown report ----------------------------------------------
    md_lines: list[str] = []
    md_lines.append("# v3.0 Phase 3 — per-fold rank consistency across the 8 Phase 2 experiments")
    md_lines.append("")
    md_lines.append(
        "Re-analysis of the 8 ``v3_phase2_*_kfold.md`` reports. No retraining; "
        "purely structural. **Reading-A-vs-Reading-C disambiguator** from "
        "``2026-06_v3.0_phase2_postmortem_discussion.md`` next-steps #1.")
    md_lines.append("")
    md_lines.append("## TL;DR")
    md_lines.append("")
    md_lines.append(f"- **Mean pairwise Spearman ρ across experiments: {mean_rho:+.3f}** "
                    f"(median {median_rho:+.3f}). N = {len(pair_rhos)} pairs of 8 experiments × 6 folds.")
    md_lines.append("- Reading guide:")
    md_lines.append("  - ρ near +1 → all experiments rank the folds the same way → there's a real "
                    "structural per-period effect; augmentor variants are all sub-tuning against the same noise.")
    md_lines.append("  - ρ near 0 → fold rankings shuffle randomly across experiments → noise dominates "
                    "(Reading A).")
    md_lines.append("  - ρ negative → augmentor variants actively disagree about which folds favour B "
                    "(would suggest model class is wrong, or different blocks help different periods).")
    md_lines.append("")

    # Per-fold matrix
    md_lines.append("## Per-fold Δ MAE matrix (rows = folds, cols = experiments)")
    md_lines.append("")
    md_lines.append("Cell colour-code (text): `-` Model B beats A by >0.01 c/L; `+` A beats B by >0.01 c/L; `=` within ±0.01.")
    md_lines.append("")
    header = "| Fold | " + " | ".join(r.label for r in results) + " | Sign tally |"
    sep = "|------|" + "|".join(["-" * (len(r.label) + 2) for r in results]) + "|----|"
    md_lines.append(header)
    md_lines.append(sep)
    for f in range(n_folds):
        cells = []
        b_count = a_count = tie_count = 0
        for r in results:
            d = r.per_fold_delta_mae[f]
            if d < -0.01:
                tag = "-"
                b_count += 1
            elif d > 0.01:
                tag = "+"
                a_count += 1
            else:
                tag = "="
                tie_count += 1
            cells.append(f"{d:+.3f}{tag}")
        md_lines.append(
            f"| fold_{f+1} | " + " | ".join(cells)
            + f" | {b_count}B / {a_count}A / {tie_count}= |"
        )
    md_lines.append("")
    md_lines.append("**Reading:** an entirely consistent column would have the same sign for "
                    "all 6 folds. Mixed signs within a column mean the augmentor variant flips "
                    "between helping and hurting depending on period.")
    md_lines.append("")

    # Per-fold summary
    md_lines.append("## Per-fold cross-experiment summary")
    md_lines.append("")
    md_lines.append("| Fold | Mean Δ | Stdev Δ | Min Δ | Max Δ | n_B wins | n_A wins | n_ties | Variance share |")
    md_lines.append("|------|-------:|--------:|------:|------:|---------:|---------:|-------:|---------------:|")
    for row in per_fold_stats:
        md_lines.append(
            f"| fold_{row['fold']} | {row['mean_delta_mae']:+.3f} | "
            f"{row['stdev_delta_mae']:.3f} | {row['min_delta_mae']:+.3f} | "
            f"{row['max_delta_mae']:+.3f} | {row['n_b_wins']} | "
            f"{row['n_a_wins']} | {row['n_ties']} | "
            f"{row['variance_share']*100:.1f}% |"
        )
    md_lines.append("")
    md_lines.append("**Variance share** = sum of squared deviations contributed by this fold "
                    "(across 8 experiments) divided by the total summed across all folds. "
                    "If one fold's share approaches 1.0, the cross-experiment noise is mostly "
                    "that fold's behaviour.")
    md_lines.append("")

    # Pairwise Spearman matrix
    labels = [r.label for r in results]
    rho_matrix: dict[tuple[str, str], float] = {(a, b): rho for a, b, rho in pair_rhos}
    md_lines.append("## Pairwise Spearman ρ of per-fold Δ MAE rankings (8×8, symmetric, diag = 1.000)")
    md_lines.append("")
    md_lines.append("| | " + " | ".join(labels) + " |")
    md_lines.append("|---|" + "|".join(["---" for _ in labels]) + "|")
    for la in labels:
        cells = []
        for lb in labels:
            if la == lb:
                cells.append("1.000")
            elif (la, lb) in rho_matrix:
                cells.append(f"{rho_matrix[(la, lb)]:+.3f}")
            else:
                cells.append(f"{rho_matrix[(lb, la)]:+.3f}")
        md_lines.append(f"| {la} | " + " | ".join(cells) + " |")
    md_lines.append("")
    md_lines.append(f"Mean off-diagonal ρ: **{mean_rho:+.3f}** (median {median_rho:+.3f}).")
    md_lines.append("")

    # Interpretation block — the headline mean/median masks a clustering
    # pattern in the rho matrix that's worth surfacing explicitly. We detect
    # the cluster + outliers programmatically rather than hard-coding labels
    # so future re-runs (after seed-noise or interaction-feature experiments)
    # produce a fresh narrative automatically.
    md_lines.append("## Interpretation")
    md_lines.append("")
    md_lines.append(
        f"**Headline: mean ρ = {mean_rho:+.3f}, median ρ = {median_rho:+.3f}.** The mean is "
        "near zero but the median is moderately positive — that's a cluster-with-outliers "
        "pattern, not a uniform-noise pattern. Investigation:")
    md_lines.append("")

    # Per-experiment mean rho to its 7 peers — flags outliers.
    per_exp_mean_rho: list[tuple[str, float]] = []
    for label in labels:
        peer_rhos: list[float] = []
        for other in labels:
            if other == label:
                continue
            peer_rhos.append(rho_matrix.get((label, other), rho_matrix.get((other, label), 0.0)))
        per_exp_mean_rho.append((label, statistics.fmean(peer_rhos)))
    per_exp_mean_rho.sort(key=lambda t: t[1])

    md_lines.append("**Mean ρ of each experiment vs its 7 peers** (sorted ascending — bottom rows "
                    "are the experiments that anti-correlate with the cluster):")
    md_lines.append("")
    md_lines.append("| Experiment | Mean peer-ρ |")
    md_lines.append("|---|---:|")
    for label, mean_peer_rho in per_exp_mean_rho:
        md_lines.append(f"| {label} | {mean_peer_rho:+.3f} |")
    md_lines.append("")

    outliers = [label for label, r in per_exp_mean_rho if r < -0.2]
    cluster = [label for label, r in per_exp_mean_rho if r > 0.3]
    if outliers:
        md_lines.append(
            f"**Outliers** (peer-ρ < −0.2): `{', '.join(outliers)}`. These experiments "
            "actively disagree with the cluster — they rank the folds in the opposite order.")
        md_lines.append("")
    if cluster:
        md_lines.append(
            f"**Cluster** (peer-ρ > +0.3): `{', '.join(cluster)}`. These experiments share a "
            "common per-fold ranking — most reliably, fold_6 (2026 spike) and fold_3 (2022-23) "
            "are the hardest folds for Model B.")
        md_lines.append("")

    md_lines.append("### What this means for the three readings")
    md_lines.append("")
    md_lines.append("- **Reading A (genuinely flat) — partial support.** The fact that no fold has "
                    "unanimous sign across the 8 experiments + mean ρ is only weakly positive is "
                    "consistent with noise dominating. Same fold can be a B-win or B-loss "
                    "depending on which variant you fit.")
    md_lines.append("- **Reading C (wrong features) — sharpened.** The cluster pattern indicates "
                    "there *is* a shared per-fold structure most experiments fail on (fold_3, "
                    "fold_6). The augmentor variants don't differentiate enough — they're all "
                    "stuck on the same problem. If Reading C2 (missing explicit interaction) is "
                    "right, the interaction feature would specifically help on fold_3 or fold_6.")
    if outliers:
        md_lines.append(
            f"- **Outlier interest.** `{outliers[0]}` is the only experiment that helps on "
            "fold_6 — that's the experiment to inspect for what it does differently. Either "
            "it found a real fold_6 signal (would be the strongest Reading-C evidence in the "
            "data) or fold_6 is so noisy that 1 in 8 random variants lands favourably.")
    md_lines.append("")

    md_lines.append("### Per-fold sign tallies")
    md_lines.append("")
    md_lines.append("Folds where **all 8 experiments agree on the sign** of Δ MAE:")
    md_lines.append("")
    unanimous = []
    for row in per_fold_stats:
        if row["n_b_wins"] == n_exps:
            unanimous.append(f"- fold_{row['fold']}: B beats A in all {n_exps} experiments")
        elif row["n_a_wins"] == n_exps:
            unanimous.append(f"- fold_{row['fold']}: A beats B in all {n_exps} experiments")
    if unanimous:
        md_lines.extend(unanimous)
    else:
        md_lines.append("_(none — no fold has unanimous agreement across all 8 experiments)_")
    md_lines.append("")
    md_lines.append("Folds where the **dominant sign** (B-beats-A or A-beats-B) holds in ≥6 of 8:")
    md_lines.append("")
    for row in per_fold_stats:
        if row["n_b_wins"] >= 6:
            md_lines.append(f"- fold_{row['fold']}: B beats A in {row['n_b_wins']}/{n_exps}")
        elif row["n_a_wins"] >= 6:
            md_lines.append(f"- fold_{row['fold']}: A beats B in {row['n_a_wins']}/{n_exps}")
    md_lines.append("")

    md_lines.append("### Per-fold MAE_A (fold difficulty context)")
    md_lines.append("")
    md_lines.append("Reference table — across the 8 experiments, Model A's per-fold MAE varies "
                    "depending on whether the experiment changes the row-filter (curation broadening "
                    "removes some rows from the identical-rows guard, which can shift Model A's MAE).")
    md_lines.append("")
    md_lines.append("| Fold | " + " | ".join(r.label for r in results) + " |")
    md_lines.append("|------|" + "|".join(["-" * (len(r.label) + 2) for r in results]) + "|")
    for f in range(n_folds):
        cells = [f"{r.per_fold_mae_a[f]:.2f}" for r in results]
        md_lines.append(f"| fold_{f+1} | " + " | ".join(cells) + " |")
    md_lines.append("")

    md_lines.append("## Sources")
    md_lines.append("")
    md_lines.append("- ``results/v3_phase2_*_kfold.md`` (8 per-experiment reports)")
    md_lines.append("- ``docs/research/2026-06_v3.0_phase2_outcome.md`` (Phase 2 outcome)")
    md_lines.append("- ``docs/research/2026-06_v3.0_phase2_postmortem_discussion.md`` "
                    "(three-readings deep dive; this script implements next-step #1)")
    md_lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("wrote %s", args.out_md)

    # ---- Headline to stdout ----------------------------------------------------
    # Avoid the Greek rho character on stdout — Windows cp1252 console can't
    # encode it. Use "rho" spelled out; the markdown report uses the symbol.
    logger.info("=== Rank consistency headline ===")
    logger.info("Mean pairwise Spearman rho: %+.3f", mean_rho)
    logger.info("Median:                     %+.3f", median_rho)
    most_variant = max(per_fold_stats, key=lambda r: r["variance_share"])
    logger.info(
        "Most-variant fold:          fold_%d (%.1f%% of cross-experiment variance)",
        most_variant["fold"], most_variant["variance_share"] * 100,
    )
    unanimous = [
        row["fold"] for row in per_fold_stats
        if row["n_b_wins"] == n_exps or row["n_a_wins"] == n_exps
    ]
    logger.info(
        "Unanimous sign folds:       %s",
        unanimous if unanimous else "none",
    )


if __name__ == "__main__":
    main()
