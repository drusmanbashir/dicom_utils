#!/usr/bin/env python3
import argparse
from pathlib import Path

from dicom_utils.drli_helper import ConvertDRLIToSITK


def main(args):
    converter = ConvertDRLIToSITK(
        dicom_folder=Path(args.dicom_folder),
        output_folder=Path(args.output_folder),
        ext=args.ext,
    )
    converter.process(
        target_label=args.target_label,
        overwrite=args.overwrite,
        debug=args.debug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a DRLI DICOM dataset into image and mask SimpleITK files.")
    parser.add_argument("--dicom-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--ext", default="nii.gz")
    parser.add_argument("--target-label", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_known_args()[0]
    main(args)
