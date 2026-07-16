#!/usr/bin/env python3
import argparse
from pathlib import Path

from dicom_utils.dcm_to_sitk import DCMDatasetToSITK


def main(args):
    output_folder = Path(args.output_folder) if args.output_folder else None
    starting_ind = None if args.use_folder_names else args.starting_ind
    converter = DCMDatasetToSITK(
        dataset_name=args.dataset_name,
        input_folder=Path(args.input_folder),
        output_folder=output_folder,
        starting_ind=starting_ind,
        tags=args.tags,
        max_series_per_case=args.max_series_per_case,
        min_files_per_series=args.min_files_per_series,
        sitk_ext=args.sitk_ext,
        include_modalities=args.include_modalities,
        exclude_image_type_values=args.exclude_image_type_values,
    )
    converter.create_cases_summary()
    converter.save_cases_summary()
    if not args.summary_only:
        converter.process_all_cases(
            sitk_ext=args.sitk_ext,
            debug=args.debug,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a DICOM dataset folder into SimpleITK volumes.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder")
    parser.add_argument("--starting-ind", type=int, default=0)
    parser.add_argument("--use-folder-names", action="store_true")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--max-series-per-case", type=int, default=3)
    parser.add_argument("--min-files-per-series", type=int, default=50)
    parser.add_argument("--sitk-ext", default=".nii.gz")
    parser.add_argument("--include-modalities", nargs="*")
    parser.add_argument("--exclude-image-type-values", nargs="*")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_known_args()[0]
    main(args)
