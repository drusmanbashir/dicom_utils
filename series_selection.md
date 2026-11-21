# Series selection (DICOM → NIfTI)

Which CT series of a study become NIfTI files. The code is `dicom_utils/series_selection.py`, called by `DCMCaseToSITK.process`. Operator rulings of 2026-11-21 on the DAR043 delivery (108 cases, 1,278 series).

## Rule
Per study date, convert one series for each distinct (body region, contrast phase, kernel class, metal-artifact reduction):

1. **Candidates:** the series folder holds one SeriesInstanceUID, the modality is in `include_modalities`, the ImageType has none of `exclude_image_type_values`, and the folder has at least `min_files_per_series` files.
2. **Axial only.** The ImageOrientationPatient slice normal has |z| > 0.8, and SeriesDescription has no `cor` or `sag` word. Coronal and sagittal reformats are never converted.
3. **Body region from z coverage.** Two series are the same region when their z ranges (min to max ImagePositionPatient z over all images) overlap by ≥ 80% of the shorter one. Regions are the connected groups.
   - A lung reconstruction inside a CAP is the same region as the CAP. A separate thorax scan and a separate abdomen scan are two regions, even though they overlap at the diaphragm.
   - Descriptions are not used for region. Most DAR043 descriptions carry no anatomy ("Source", "1/1 Soft", "3/2 Lung", "_LCAD", "Body Body 1.0"), and "Lung" there means a lung-kernel reconstruction, not a separate scan.
   - DAR043 result: 99 cases have one region and 9 have two (e.g. Thorax + Abdomen, or two "Body" acquisitions).
4. **Contrast phase.** In each region, routine (portal-venous) series are kept. Arterial (`arterial`, `art`, `angio…`, `cta`) and unenhanced (`non contrast`, `non con`, `unenhanced`, `plain`, `pre contrast`, `nc`) series are dropped when the region has a routine series. A region with no routine series keeps its arterial or unenhanced series (e.g. an unenhanced-only CT chest).
5. **Kernel class.** Both are kept: `soft` (default) and `lung`. A series is `lung` when SeriesDescription contains `lung`, `hrct` or `lcad`, or its kernel is LUNG, BONE, Y-A..D (Philips), Bl57-type (Siemens lung), or B/I/Br/Bf 60–99 (Siemens sharp). BODY_SHARP is a body kernel, so `soft`.
6. **Metal-artifact reduction.** Both are kept: O-MAR/iMAR versions and plain versions ("Non O-MAR" counts as plain).
7. **Thickness, within one (region, phase, kernel, MAR) group.** One series: if any is thinner than 0.8 mm, the thickest of those; otherwise the thinnest. Ties go to the series with more files.

## Output names
`<case_id>_<tags>.nii.gz`. The SDE wrapper uses tags `StudyDate` and `SeriesDescription`; the description is lower-cased with only a–z and 0–9 kept, e.g. `test_000009_20250103_cap15mm.nii.gz`. When two selected series would get the same name, each gets `_s<SeriesNumber>`, e.g. `..._bodybody10_s4.nii.gz` and `..._bodybody10_s7.nii.gz`.

`cases_summary.csv` gets one row per case, with `;`-joined `output`, `SeriesInstanceUID`, `SeriesDescription`, `SliceThickness` and `source_file_count` in the same order.

## DAR043 run (2026-11-21, CPU VM, 1 h 55 min)
211 files from 108 cases, 0 failed. Files per case: 1 file 16 cases, 2 files 81, 3 files 11. Images per series: soft 164–1017 (median 516, 122 series), lung 66–672 (median 312, 89 series). 26 GB. Six series raised ITK's "non uniform sampling or missing slices" warning (`run_1816.log`).

## Limits
`max_series_per_case` caps study dates per case, not files. A case can now give several files per study.
