"""Re-extract correct headline metrics + rewrite pr_c_overnight_summary.md.

The orchestrator's `_extract_headline_metrics` matched any `| test_normal |`
line in a comparison.md, so it picked up the LAST one — the B-vs-B' table
— instead of the FIRST one (A vs B headline). Rewrite the summary with
the correct numbers + the per-experiment failure root causes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

EXPERIMENTS = (
    ("e1_dss_temporal", "Move 13 DSS variables to temporal pass (orig §7.7.2 motivation, unblocked by #99 / #100)"),
    ("e2_gcp_temporal", "Move GCP direct + GCP-internal PRESETs (9 vars) to temporal (unblocked by #91 Stage 2)"),
    ("e3_combined_temporal", "Combine: DSS + GCP both move to temporal (kitchen-sink)"),
    ("e4_new_erp_density_plus_curation",
     "Adopt PR #97's new ERP.population_density_per_km2 + broaden SA2_COLUMNS with PR A's 5 unmodeled candidates + the new density (temporal pass = PR B baseline)"),
)


def extract_first_a_vs_b(comparison_path: Path) -> dict:
    """Pull the FIRST `## Headline (overall) — A vs B` table's metrics.

    Returns dict with test_normal_{mae_a, mae_b, delta_mae} +
    test_crisis_{mae_a, mae_b, delta_mae}, or {} on parse failure.
    """
    text = comparison_path.read_text(encoding="utf-8")
    # Find the A vs B table heading, take the next "test_normal" + "test_crisis" rows.
    avb_match = re.search(r"## Headline \(overall\) [—-] A vs B(.*?)## Headline", text, re.DOTALL)
    if not avb_match:
        return {}
    block = avb_match.group(1)
    out: dict = {}
    for fold in ("test_normal", "test_crisis"):
        m = re.search(rf"\|\s*{fold}\s*\|([^\|]+)\|([^\|]+)\|([^\|]+)\|([^\|]+)\|", block)
        if not m:
            continue
        try:
            out[f"{fold}_n"] = int(m.group(1).strip().replace(",", ""))
            out[f"{fold}_mae_a"] = float(m.group(2).strip())
            out[f"{fold}_mae_b"] = float(m.group(3).strip())
            out[f"{fold}_delta_mae"] = float(m.group(4).strip())
        except (IndexError, ValueError):
            pass
    return out


def main() -> None:
    # Load the raw metrics JSON (still useful for error messages + wall_clock_min).
    raw = json.loads((RESULTS_DIR / "pr_c_overnight_metrics.json").read_text(encoding="utf-8"))

    # Re-extract baseline from results/comparison.md (the committed PR B state).
    baseline_path = RESULTS_DIR / "comparison.md"
    baseline = extract_first_a_vs_b(baseline_path)
    print(f"baseline (PR B): {baseline}")

    # Re-extract each experiment's metrics from its own comparison.md.
    fixed_metrics: dict = {}
    for name, _desc in EXPERIMENTS:
        exp_path = RESULTS_DIR / f"pr_c_{name}_comparison.md"
        if exp_path.exists():
            m = extract_first_a_vs_b(exp_path)
            # Preserve wall_clock_min + error from the original raw file.
            old = raw["experiments"].get(name, {})
            for k in ("wall_clock_min", "error"):
                if k in old:
                    m[k] = old[k]
            fixed_metrics[name] = m
        else:
            # Failed experiment — preserve error
            fixed_metrics[name] = raw["experiments"].get(name, {"error": "no comparison file written"})

    print(f"\nfixed experiment metrics: {json.dumps(fixed_metrics, indent=2)}")

    # Update raw metrics file
    raw_fixed = {"baseline": baseline, "experiments": fixed_metrics}
    (RESULTS_DIR / "pr_c_overnight_metrics.json").write_text(
        json.dumps(raw_fixed, indent=2), encoding="utf-8"
    )

    # Write a much-improved summary.
    lines = [
        "# PR C overnight experiment summary",
        "",
        "Run via `tools/research/pr_c_overnight_runner.py` against augmentor",
        "pin `762a6a0f` (post our #91 Stage 2 + #99 fixes + the new ERP",
        "density column). Numbers re-extracted by",
        "`tools/research/pr_c_fix_summary.py` after a bug in the original",
        "orchestrator's extractor was discovered (it had matched the B-vs-B'",
        "table instead of the A-vs-B headline).",
        "",
        "## TL;DR",
        "",
        "**1 of 4 experiments completed; 3 failed with distinct upstream issues.**",
        "The one that ran (E4 — curation broadening + new ERP density) shows the",
        "now-familiar pattern: it traded test_normal Δ MAE for a substantially better",
        "test_crisis Δ MAE. The crisis-fold improvement is the larger gross movement.",
        "",
        "## Baseline (committed PR B, headline A vs B)",
        "",
        "| Fold | MAE A | MAE B | Δ MAE |",
        "|------|------:|------:|------:|",
    ]
    for fold in ("test_normal", "test_crisis"):
        a = baseline.get(f"{fold}_mae_a", float("nan"))
        b = baseline.get(f"{fold}_mae_b", float("nan"))
        d = baseline.get(f"{fold}_delta_mae", float("nan"))
        lines.append(f"| {fold} | {a:.3f} | {b:.3f} | **{d:+.3f}** |")

    lines.extend([
        "",
        "## Experiments",
        "",
        "Δ MAE: negative = Model B beats Model A. \"vs baseline\" subtracts the",
        "baseline Δ MAE — negative = this experiment improved on PR B.",
        "",
        "| Experiment | test_normal Δ MAE | vs baseline | test_crisis Δ MAE | vs baseline | wall-clock | status |",
        "|------------|------------------:|------------:|------------------:|------------:|-----------:|--------|",
    ])
    for name, _desc in EXPERIMENTS:
        m = fixed_metrics.get(name, {})
        if "error" in m and "test_normal_delta_mae" not in m:
            lines.append(
                f"| **{name}** | _failed_ | _failed_ | _failed_ | _failed_ | _failed_ "
                f"| ❌ {m['error'][:60]}{'…' if len(m['error']) > 60 else ''} |"
            )
            continue
        d_normal = m.get("test_normal_delta_mae", float("nan"))
        d_crisis = m.get("test_crisis_delta_mae", float("nan"))
        bn = baseline.get("test_normal_delta_mae", float("nan"))
        bc = baseline.get("test_crisis_delta_mae", float("nan"))
        diff_normal = d_normal - bn
        diff_crisis = d_crisis - bc
        wall = m.get("wall_clock_min", float("nan"))
        lines.append(
            f"| **{name}** "
            f"| {d_normal:+.3f} "
            f"| {diff_normal:+.3f} "
            f"| {d_crisis:+.3f} "
            f"| {diff_crisis:+.3f} "
            f"| {wall:.1f} min "
            f"| ✅ |"
        )

    lines.extend([
        "",
        "## Failure modes (E1, E2, E3) — three different root causes",
        "",
        "### E1 — DSS temporal — schema-drift across DSS quarterly releases",
        "",
        "```",
        "ValueError: Dataset 'dss_payments' doesn't expose columns",
        "  ['family_tax_benefit_a_recipients', 'family_tax_benefit_b_recipients'].",
        "```",
        "",
        "The augmentor's DSS parser validates that every requested column exists",
        "in **every** release that temporal-mode might resolve to. The FTB-A/B",
        "columns exist in current quarters (we use them cross-sectional with no",
        "issue) but appear to be missing or differently-named in older quarterly",
        "releases (the parser fix from #100 added historical support but didn't",
        "harmonise the column schema across releases).",
        "",
        "**Cheap fix:** drop FTB-A/B from the temporal DSS set; keep them",
        "cross-sectional (latest quarter only). The other 11 DSS columns may",
        "still be temporal-eligible. Worth filing as an augmentor issue: \"DSS",
        "temporal mode rejects request when any column is absent from any",
        "historical release; consider a per-release intersection or per-row",
        "graceful null instead.\"",
        "",
        "### E2 / E3 — GCP temporal — `cannot reindex on an axis with duplicate labels`",
        "",
        "This is the same upstream PRESET-collision class we worked around in",
        "`src/fuel_pred/build/enrich_census.py` with `_split_for_preset_collision`.",
        "That splitter only fires for the cross-sectional path. The temporal",
        "pass (`enrich_panel_temporal._augment`) sends all variables in a single",
        "`Pipeline.augment` call without the same defence, and hits the duplicate-",
        "label error when GCP-internal PRESETs collide with direct G## refs.",
        "",
        "**Cheap fix:** extract `_split_for_preset_collision` to a shared helper",
        "and apply it in `enrich_panel_temporal._augment` too (two passes,",
        "column-wise merge). E3 fails for the same reason E2 does (E3 includes",
        "the GCP family).",
        "",
        "### Aside: upstream regression `compute_sa2_areas_km2` (filed as #101)",
        "",
        "On first launch all four experiments crashed at `Pipeline.create()`",
        "before any pipeline stage even started, with",
        "`AttributeError: 'NoneType' object has no attribute 'area'`. This was",
        "an upstream regression introduced by PR #97 (the new ERP density column)",
        "— `compute_sa2_areas_km2` iterates over (code, geometry) tuples and",
        "assumes geometry is never None. Filed as augmentor #101 with the trivial",
        "one-line fix. Workaround monkey-patch installed at the top of",
        "`pr_c_overnight_runner.py`; remove once upstream lands.",
        "",
        "## E4 result interpretation (the one that ran)",
        "",
        "Setup: same temporal pass as PR B (SEIFA + ERP-total only) + added",
        "`ERP.population_density_per_km2` to cross-sectional + grew `SA2_COLUMNS`",
        "from 15 to 21 (added: erp_population_65_plus, erp_median_age,",
        "pct_age_pension_recipients, pct_jobseeker_recipients, welfare_density_index,",
        "erp_population_density_per_km2 — i.e. PR A's 5 unmodeled candidates +",
        "the brand-new density column).",
        "",
        "Result: **test_crisis Δ MAE -0.603 (up from PR B's -0.321; +0.282 c/L",
        "improvement)** but **test_normal Δ MAE -0.088 (down from PR B's -0.239;",
        "-0.151 c/L regression)**. Mixed.",
        "",
        "Same pattern we saw in v1.5 (the 31-col broadening that beat val but lost",
        "test) and in PR B (temporal-SEIFA traded normal-fold accuracy for crisis-",
        "fold gain). The augmentor surface seems to have a real trade-off here —",
        "more SA2 features help on the OOD crisis fold but hurt the in-",
        "distribution test_normal fold.",
        "",
        "Open question: is the test_crisis lift dominated by",
        "`erp_population_density_per_km2` specifically, or is it the curation",
        "broadening? Worth a follow-up ablation (E4-density-only vs E4-curation-",
        "only) to attribute.",
        "",
        "## Per-experiment artefacts",
        "",
        "| Experiment | Comparison report | Features parquet | Models dir |",
        "|------------|-------------------|------------------|------------|",
    ])
    for name, _desc in EXPERIMENTS:
        comp = RESULTS_DIR / f"pr_c_{name}_comparison.md"
        feat = REPO_ROOT / "data" / "processed" / f"features_{name}.parquet"
        mdir = REPO_ROOT / f"models_{name}"
        comp_exists = "✅" if comp.exists() else "—"
        feat_exists = "✅" if feat.exists() else "—"
        mdir_exists = "✅" if mdir.exists() else "—"
        lines.append(
            f"| {name} "
            f"| `results/pr_c_{name}_comparison.md` {comp_exists} "
            f"| `data/processed/features_{name}.parquet` {feat_exists} "
            f"| `models_{name}/` {mdir_exists} |"
        )

    lines.extend([
        "",
        "## Suggested next steps",
        "",
        "1. **Don't merge anything from this branch yet** — the PR C results don't",
        "   beat PR B on the primary fold (test_normal). The crisis-fold lift in",
        "   E4 is interesting but needs ablation before committing.",
        "2. **File augmentor issue** for the DSS cross-release schema-drift",
        "   (E1 cause). Either harmonise the schema or accept per-release",
        "   intersections.",
        "3. **Fix locally**: factor `_split_for_preset_collision` out of",
        "   `enrich_census.py` into a shared helper that `enrich_panel_temporal`",
        "   also uses. Then re-run E2/E3.",
        "4. **Wait for augmentor #101 fix** before re-running anything against",
        "   future upstream commits; ours installs the workaround inline but",
        "   that should go upstream.",
        "5. **E4 ablation experiment** — split into E4a (just the new density",
        "   column, no curation broadening) and E4b (curation broadening only,",
        "   no new density). Attributes the +0.282 c/L crisis-fold gain to",
        "   whichever component is responsible.",
        "",
        "## Experiment descriptions",
        "",
    ])
    for name, desc in EXPERIMENTS:
        lines.append(f"- **{name}** — {desc}")
    lines.append("")

    (RESULTS_DIR / "pr_c_overnight_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\nwrote updated summary to results/pr_c_overnight_summary.md")


if __name__ == "__main__":
    main()
