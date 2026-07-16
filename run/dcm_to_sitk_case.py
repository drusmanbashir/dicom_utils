#!/usr/bin/env python3
import argparse
from pathlib import Path

from dicom_utils.dcm_to_sitk import DCMCaseToSITK


def main(args):
    study_date = args.study_date or None
    exclude_vals = [] if args.allow_derived else args.exclude_image_type_values
    converter = DCMCaseToSITK(
        dataset_name=args.dataset_name,
        case_folder=Path(args.case_folder),
        output_folder=Path(args.output_folder),
        case_id=args.case_id,
        tags=args.tags,
        max_series_per_case=args.max_series_per_case,
        min_files_per_series=args.min_files_per_series,
        sitk_ext=args.sitk_ext,
        include_modalities=args.include_modalities,
        exclude_image_type_values=exclude_vals,
        study_date=study_date,
    )
    converter.process(overwrite=args.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert one DICOM case folder into SimpleITK volumes.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--case-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--study-date", default="")
    parser.add_argument("--max-series-per-case", type=int, default=2)
    parser.add_argument("--min-files-per-series", type=int, default=1)
    parser.add_argument("--sitk-ext", default=".nii.gz")
    parser.add_argument("--include-modalities", nargs="*")
    parser.add_argument("--exclude-image-type-values", nargs="*")
    parser.add_argument("--allow-derived", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_known_args()[0]
    main(args)
