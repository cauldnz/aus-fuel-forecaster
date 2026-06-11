# v6 — intrinsic dimensionality of the full augmentor surface

Self-prediction / PCA analysis of the **37-column** full augmentor surface (`data/interim/stations.parquet`), 580 unique SA2 profiles. A property of the augmentor data itself, independent of the fuel problem.

## 1. Intrinsic dimensionality

| Variance explained | Components (of 37) |
|---|---:|
| 80% | 4 |
| 90% | 6 |
| 95% | 10 |
| 99% | 20 |

Top-8 explained-variance ratio: 0.399, 0.240, 0.129, 0.070, 0.046, 0.026, 0.012, 0.012.

## 2. Leave-one-column-out predictability

- **Mean R² = 0.919** (GBM), median 0.936; Ridge mean 0.939
- GBM R² > 0.5 (majority-predictable): **37/37**
- GBM R² > 0.9 (near-perfectly reconstructable): **26/37**

| Column | Ridge R² | GBM R² |
|--------|---------:|-------:|
| sa2_dss_family_tax_benefit_a_recipients | 0.998 | 0.989 |
| sa2_dss_family_tax_benefit_b_recipients | 0.998 | 0.989 |
| sa2_seifa_irsad_score | 0.996 | 0.987 |
| sa2_median_age | 0.990 | 0.979 |
| sa2_pia_income_earners_count | 0.990 | 0.978 |
| sa2_total_population | 0.990 | 0.977 |
| sa2_erp_median_age | 0.985 | 0.976 |
| sa2_erp_population_total | 0.981 | 0.975 |
| sa2_dss_age_pension_recipients | 0.981 | 0.972 |
| sa2_seifa_ieo_score | 0.986 | 0.971 |
| sa2_seifa_irsd_score | 0.977 | 0.966 |
| sa2_erp_population_65_plus | 0.988 | 0.965 |
| sa2_dss_jobseeker_payment_recipients | 0.974 | 0.960 |
| sa2_median_household_income_weekly | 0.956 | 0.956 |
| sa2_seifa_ier_score | 0.983 | 0.952 |
| sa2_welfare_density_index | 0.983 | 0.949 |
| sa2_pia_median_age_of_earners | 0.958 | 0.939 |
| sa2_dss_parenting_payment_single_recipients | 0.980 | 0.938 |
| sa2_dss_carer_allowance_recipients | 0.989 | 0.936 |
| sa2_dss_disability_support_pension_recipients | 0.950 | 0.934 |
| sa2_pia_median_total_income | 0.923 | 0.934 |
| sa2_pct_aged_65_plus | 0.956 | 0.929 |
| sa2_motor_vehicles_per_dwelling | 0.934 | 0.921 |
| sa2_pct_jobseeker_recipients | 0.954 | 0.915 |
| sa2_dss_commonwealth_seniors_health_card_recipients | 0.943 | 0.911 |
| sa2_dss_youth_allowance_other_recipients | 0.917 | 0.908 |
| sa2_pct_renters | 0.935 | 0.900 |
| sa2_dss_carer_payment_recipients | 0.981 | 0.896 |
| sa2_pia_mean_total_income | 0.883 | 0.883 |
| sa2_dss_commonwealth_rent_assistance_recipients | 0.920 | 0.881 |
| sa2_pct_age_pension_recipients | 0.907 | 0.867 |
| sa2_pct_drive_to_work | 0.833 | 0.859 |
| sa2_pct_employed_full_time | 0.724 | 0.846 |
| sa2_dss_parenting_payment_partnered_recipients | 0.904 | 0.838 |
| sa2_pct_one_parent_family | 0.829 | 0.827 |
| sa2_erp_population_density_per_km2 | 0.794 | 0.750 |
| sa2_dss_youth_allowance_student_and_apprentice_recipients | 0.781 | 0.667 |

## 3. Correlation + latent axes

- Mean max |r| to any other column: **0.885**
- Columns with a >0.7 correlated partner: 35/37; >0.9: 21/37

**Top principal-component loadings (the latent axes):**

- **PC1 (40% var):** +0.23 dss_jobseeker_payment, +0.23 dss_disability_support_pension, +0.23 dss_family_tax_benefit_b, +0.23 dss_youth_allowance_other, +0.23 dss_commonwealth_rent_assistance, +0.22 dss_family_tax_benefit_a
- **PC2 (24% var):** -0.28 pia_income_earners_count, -0.27 erp_population_total, +0.26 erp_median_age, -0.26 total_population, +0.25 median_age, +0.24 pct_aged_65_plus
- **PC3 (13% var):** +0.38 dss_commonwealth_seniors_health_card, +0.37 erp_population_65_plus, +0.28 pia_median_age_of_earners, +0.27 dss_age_pension, -0.25 pct_renters, +0.24 seifa_ier_score
- **PC4 (7% var):** -0.47 motor_vehicles_per_dwelling, -0.34 seifa_ier_score, +0.30 erp_population_density_per_km2, +0.30 pct_renters, +0.23 pct_aged_65_plus, -0.22 pct_drive_to_work

## Sources

- `tools/research/v6_augmentor_pca.py` — this script
- `data/interim/stations.parquet` — full materialized augmentor surface
- Complements the fuel-utility null in `docs/research/2026-06_v3.0_phase3_closing_summary.md`
