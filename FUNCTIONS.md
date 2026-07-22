# FUNCTIONS.md

Scope: only code under `run/`, `cli/`, or `tools/` belongs here.

`run/` now contains thin CLI entry points for the reusable conversion helpers in `dicom_utils/`.

## `run/dcm_to_sitk_dataset.py`

- Calls `dicom_utils.dcm_to_sitk.DCMDatasetToSITK`.
- Creates or extends `cases_summary.csv`, saves it, and optionally processes every case into SITK outputs.
- Key args:
  - `--dataset-name`
  - `--input-folder`
  - `--output-folder`
  - `--use-folder-names`
  - `--tags`
  - `--summary-only`
  - `--overwrite`

## `run/dcm_to_sitk_case.py`

- Calls `dicom_utils.dcm_to_sitk.DCMCaseToSITK`.
- Converts one case folder, potentially emitting one output file per DICOM series.
- Key args:
  - `--dataset-name`
  - `--case-folder`
  - `--output-folder`
  - `--case-id`
  - `--tags` (default empty)
  - `--study-date`
  - `--include-modalities`
  - `--overwrite`
- Axial series auto-select when multiple series share a study date (1.0–1.5 mm band, else thickest axial).

## `run/drli_to_sitk.py`

- Calls `dicom_utils.drli_helper.ConvertDRLIToSITK`.
- Converts the DRLI layout into `images/` and `masks/` outputs.
- Key args:
  - `--dicom-folder`
  - `--output-folder`
  - `--ext`
  - `--target-label`
  - `--overwrite`

## `run/nifti_rgb_to_dicom.py`

- Calls `dicom_utils.sitk_to_dcm.nifti_rgb_to_dicom_series`.
- Writes an RGB NIfTI overlay back out as a DICOM series using a reference series for geometry and patient metadata.
- Key args:
  - `--nifti-path`
  - `--ref-dicom-dir`
  - `--output-dir`
  - `--series-description`
  - `--pixel-representation`

## `run/set_patient_id_from_folder.py`

- Calls `dicom_utils.metadata.dcm_id_fromfoldername`.
- Recursively rewrites `PatientID` for each DICOM file under a folder to the folder name.
- Key args:
  - `--dicom-folder`

Core package API note:
- `dicom_utils.rtstruct.RTStructStudy` provides RTSTRUCT->mask conversion on CT grid (see `README.md`).
