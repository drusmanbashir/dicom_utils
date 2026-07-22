#!/usr/bin/env python3
import argparse
from pathlib import Path

from dicom_utils.metadata import dcm_id_fromfoldername


def main(args):
    dcm_id_fromfoldername(Path(args.dicom_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set PatientID on all DICOM files under a folder to the folder name.")
    parser.add_argument("--dicom-folder", required=True)
    args = parser.parse_known_args()[0]
    main(args)
