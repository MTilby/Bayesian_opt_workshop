# Reaction datasets

These CSVs reformat numerical observations from published studies. Cite the original study when using a dataset, and cite the repository paper when using its distributed version.

| ID | File | Rows | Original source |
| --- | --- | ---: | --- |
| A1 | [Suzuki numeric](dataset_a1_aspuru_guzik_suzuki_numeric.csv) | 247 | Häse et al., Olympus, 2021. [10.1088/2632-2153/abedc8](https://doi.org/10.1088/2632-2153/abedc8) |
| B1 | [Jensen case I](dataset_b1_jensen_case_i.csv) | 96 | Reizman et al., 2016, SI Table S.1. [10.1039/C6RE00153J](https://doi.org/10.1039/C6RE00153J) |
| B2 | [Jensen case II](dataset_b2_jensen_case_ii.csv) | 96 | Same study, SI Table S.3. |
| B3 | [Jensen case III](dataset_b3_jensen_case_iii.csv) | 96 | Same study, SI Table S.5. |
| B4 | [Jensen case IV](dataset_b4_jensen_case_iv.csv) | 88 | Same study; subset distributed as Olympus `dataset_suzuki_iv`. |
| C1 | [Newman plate replicates](dataset_c1_newman_bh_plate_replicates.csv) | 288 | Cook, Clément, and Newman, 2021, SI Supporting Table 1 and plate layouts. [10.1038/s41596-020-00452-7](https://doi.org/10.1038/s41596-020-00452-7) |
| D1 | [Doyle direct arylation](dataset_d1_doyle_direct_arylation.csv) | 1,728 | Shields et al., 2021. [10.1038/s41586-021-03213-y](https://doi.org/10.1038/s41586-021-03213-y) |
| E1 | [Bourne SNAr](dataset_e1_bourne_snar.csv) | 99 | Kershaw et al., 2023, SI Tables S5–S6. [10.1016/j.cej.2022.138443](https://doi.org/10.1016/j.cej.2022.138443) |
| E2 | [Bourne Sonogashira](dataset_e2_bourne_sonogashira.csv) | 69 | Same study, SI Tables S7–S8. |

## Preparation

Columns were renamed to identify variables and units. [dataset_legend.csv](dataset_legend.csv) maps categorical labels to chemical identities; [dataset_summary.csv](dataset_summary.csv) lists inputs, responses, metadata, and source DOIs.

- B1–B3 retain ligand identity but omit the original P1/P2 precatalyst distinction. Their experiment numbering and row order differ from Summit.
- C1 retains individual plate replicates. Plate and well identifiers are metadata, not BO inputs.
- D1 replaces base, ligand, and solvent identities with labels.
- E1/E2 `is_pareto` flags are annotations based on the published Pareto tables, not measured responses or input variables.

## Machine-readable sources and reuse

- **A1/B4:** [Olympus datasets](https://github.com/the-matter-lab/olympus/tree/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets), `dataset_suzuki/data.csv` and `dataset_suzuki_iv/data.csv`. All local rows match these files apart from headers.
- **B1–B3:** [Summit datasets](https://github.com/sustainable-processes/summit/tree/1de682d05e97adcfb96cd8376e876cef2d6160d3/summit/benchmarks/data), `reizman_suzuki_case_1.csv` through `case_3.csv`. Records match after the transformations above. Summit: Felton, Rittig, and Lapkin, 2021, [10.1002/cmtd.202000051](https://doi.org/10.1002/cmtd.202000051).
- **D1:** [EDBO experiment index](https://github.com/b-shields/edbo/blob/9b41eac3f6d9e520547702fd5b0c7ef6441625a4/experiments/data/direct_arylation/experiment_index.csv). All rows match after categorical label mapping.

The links above identify specific source versions. Olympus, Summit, and EDBO have MIT licenses. E1/E2 derive from CC BY 4.0 supplementary material. See [LICENSE](../LICENSE) for attribution and license texts.
