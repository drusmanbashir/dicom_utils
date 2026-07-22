#!/usr/bin/env python3
import argparse
from pathlib import Path

from dicom_utils.sitk_to_dcm import nifti_rgb_to_dicom_series


def main(args):
    nifti_rgb_to_dicom_series(
        nifti_path=Path(args.nifti_path),
        ref_dicom_dir=Path(args.ref_dicom_dir),
        output_dir=Path(args.output_dir),
        series_description=args.series_description,
        study_instance_uid=args.study_instance_uid,
        series_instance_uid=args.series_instance_uid,
        signed=args.pixel_representation == "signed",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a 4D RGB NIfTI overlay into a DICOM series.")
    parser.add_argument("--nifti-path", required=True)
    parser.add_argument("--ref-dicom-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--series-description", default="RGB Overlay Series")
    parser.add_argument("--study-instance-uid")
    parser.add_argument("--series-instance-uid")
    parser.add_argument("--pixel-representation", choices=["signed", "unsigned"], default="signed")
    args = parser.parse_known_args()[0]
    main(args)
