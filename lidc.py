#
import configparser
import csv
import shutil
import sys
from pathlib import Path

import ipdb
import numpy as np
import SimpleITK as sitk


def _apply_runtime_compat():
    # pylidc still references deprecated NumPy scalar aliases on newer NumPy.
    if "int" not in np.__dict__:
        np.int = int
    if "bool" not in np.__dict__:
        np.bool = np.bool_
    # Python 3.12 removed SafeConfigParser; keep alias for older callers.
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser


_apply_runtime_compat()

import pylidc as pl
from pylidc.utils import consensus
from utilz.fileio import maybe_makedirs
from xnat.object_oriented import Proj


tr = ipdb.set_trace


class LIDCXNATProcessor():
    """Processes LIDC-IDRI dataset scans and uploads them to XNAT."""

    def __init__(self, imgs_fldr="/s/tmp/images", lms_fldr="/s/tmp/masks") -> None:
        if pl is None or consensus is None:
            raise ImportError("pylidc (and its dependencies, including pkg_resources/setuptools) is required for LIDCProcessor")
        self.proj_title = "lidc"
        self.proj = Proj("lidc")
        self.scans = pl.query(pl.Scan).filter()
        print(self.scans.count())
        self.imgs_fldr = Path(imgs_fldr)
        self.lm_fldr = Path(lms_fldr)
        self._scan_index_map = None
        maybe_makedirs([self.imgs_fldr, self.lm_fldr])

    def process_scan(self, scan, clevel=.35, upload_on_xnat=True):
        """Process a single LIDC scan and upload to XNAT if not already present."""
        case_id = self.get_case_id(scan)
        fn = "{0}_{1}.nii.gz".format(self.proj_title, case_id)
        fn_img = self.imgs_fldr / fn
        fn_lm = self.lm_fldr / fn
        if all([fn.exists() for fn in [fn_img, fn_lm]]):
            print("IMAGE and LM already in tmp folder")
        else:
            vol = scan.to_volume()
            lm_np = np.zeros_like(vol)
            lm_np = self.fill_lm(scan, lm_np, clevel)
            img, lm = self.scan_lm_to_nii(scan.spacings, vol, lm_np)

            sitk.WriteImage(img, fn_img)
            sitk.WriteImage(lm, fn_lm)
        if upload_on_xnat:
            self.maybe_upload_scan_rscs(scan, fn_img, fn_lm)

    def get_case_id(self, scan):
        """Extract case ID from scan patient ID."""
        case_id = scan.patient_id
        case_id = case_id.split("-")[-1]
        return case_id

    def get_filename_from_index(self, indx):
        case_id = self.get_case_id(self.scans[indx])
        return "{0}_{1}.nii.gz".format(self.proj_title, case_id)

    def get_scan_index_map(self):
        """Map output filename -> pylidc scan index."""
        if self._scan_index_map is None:
            total_scans = self.scans.count()
            self._scan_index_map = {
                self.get_filename_from_index(indx): indx
                for indx in range(total_scans)
            }
        return self._scan_index_map

    def parse_existing_entries_csv(self, csv_path):
        """Load indexes and filenames from an index CSV."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print("Index CSV not found: {0}".format(csv_path))
            return set(), set()

        indices = set()
        filenames = set()
        with csv_path.open("r", newline="") as f:
            rdr = csv.DictReader(f)
            if not rdr.fieldnames:
                return indices, filenames
            for row in rdr:
                fname = str(row.get("filename", "")).strip()
                if fname:
                    filenames.add(fname)
                val = str(row.get("index", "")).strip()
                if val:
                    indices.add(int(val))
        return indices, filenames

    def gather_existing_entries_from_csvs(self, csv_paths):
        all_indices = set()
        all_filenames = set()
        for csv_path in csv_paths or []:
            idxs, fns = self.parse_existing_entries_csv(csv_path)
            all_indices.update(idxs)
            all_filenames.update(fns)
        return all_indices, all_filenames

    def gather_existing_indices_from_csvs(self, csv_paths):
        existing, _ = self.gather_existing_entries_from_csvs(csv_paths)
        return existing

    def write_index_csv(self, dataset_root, csv_path=None):
        """
        Create a CSV with filename/case_id/index for an existing dataset folder.
        Expected folder structure: <dataset_root>/images and <dataset_root>/lms.
        """
        dataset_root = Path(dataset_root)
        images_fldr = dataset_root / "images"
        lms_fldr = dataset_root / "lms"
        if csv_path is None:
            csv_path = dataset_root / "lidc_indices.csv"
        csv_path = Path(csv_path)

        index_map = self.get_scan_index_map()
        rows = []
        for img_fn in sorted(images_fldr.glob("*.nii.gz")):
            idx = index_map.get(img_fn.name)
            lm_exists = (lms_fldr / img_fn.name).exists()
            rows.append({
                "filename": img_fn.name,
                "case_id": img_fn.name.replace(".nii.gz", "").split("_")[-1],
                "index": "" if idx is None else int(idx),
                "image_exists": 1,
                "lm_exists": int(lm_exists),
            })

        with csv_path.open("w", newline="") as f:
            wr = csv.DictWriter(
                f,
                fieldnames=["filename", "case_id", "index", "image_exists", "lm_exists"],
            )
            wr.writeheader()
            wr.writerows(rows)

        unresolved = sum(1 for r in rows if r["index"] == "")
        print(
            "Wrote {0} rows to {1}. Unresolved indexes: {2}".format(
                len(rows), csv_path, unresolved
            )
        )
        return csv_path

    def fill_lm(self, scan, lm_np, clevel=.35):
        """Fill landmark mask with consensus annotations from radiologists."""
        # clevel of 0.5 means if 1/2 ppl agree that voxel belong, it is labelled. lower threshold gives more liberal bounds
        nods = scan.cluster_annotations()
        for anns in nods:
            cmask, cbbox, lm_nps = consensus(anns, clevel=clevel)  # liberal roi
            # pad=[(20,20), (20,20), (0,0)])

            cmask_int = np.zeros(cmask.shape)

            classes = [ann.malignancy for ann in anns]
            class_ = int(np.round(np.average(classes)))
            cmask_int[cmask] = class_

            lm_np[cbbox] = cmask_int
        return lm_np

    def scan_lm_to_nii(self, spacings, vol, lm_np):
        """Convert scan volume and landmark mask to NIfTI format."""
        shape = vol.shape
        if shape[0] == shape[1] == 512:
            vol2 = vol.transpose(2, 0, 1)
            lm_np2 = lm_np.transpose(2, 0, 1)

        else:
            tr()
        img = sitk.GetImageFromArray(vol2)
        img.SetSpacing(spacings)

        lm = sitk.GetImageFromArray(lm_np2)
        lm.SetSpacing(spacings)
        return img, lm

    def maybe_create_subject(self, scan):
        """Create XNAT subject if it doesn't exist, otherwise return existing subject."""
        case_id = self.get_case_id(scan)
        subj = self.proj.subject(case_id)
        if not subj.exists():
            print("Creating subject {0}".format(case_id))
            subj.create()
        else:
            print("Using existing subject {0}".format(case_id))
        return subj

    def maybe_upload_scan_rscs(self, scan, fn_img, fn_lm):
        """Upload scan image and landmark resources to XNAT experiment."""
        subj = self.maybe_create_subject(scan)
        case_id = self.get_case_id(scan)
        exp = subj.experiment("ct_{}".format(case_id))
        if not exp.exists():
            exp.create(experiments='xnat:ctSessionData')

        print("Creating IMAGE and LM_GT resources for {0}".format(scan.patient_id))
        self.maybe_upload_rsc(exp, "IMAGE", fn_img)
        self.maybe_upload_rsc(exp, "LM_GT", fn_lm)
        print("done")

    def maybe_upload_rsc(self, exp, label, fpath):
        """Upload a file as a resource to XNAT experiment if it doesn't already exist."""

        rsc = exp.resource(label)
        if not rsc.exists():
            print("Uploading {0} as resource label {1}".format(fpath, label))
            rsc.file(fpath.name).put(fpath)
        else:
            print("File already in resource {0}".format(label))


# %%
# SECTION:-------------------- COLLATE DONLOADED NII FILES FROM XNAT--------------------------------------------------------------------------------------
if __name__ == '__main__':

    # %%
    img_fns = list(src_fldr.rglob("*IMAGE*/*.nii.gz"))
    lm_fns = list(src_fldr.rglob("*LM*/*.nii.gz"))
    # %%
    desti_fldr = lms_fldr
    for fn in lm_fns:
        fn_neo = desti_fldr / fn.name
        print("{0}   ---->   {1}".format(fn, fn_neo))
        shutil.move(fn, fn_neo)
    # %%
    desti_fldr = img_fldr
    for fn in img_fns:
        fn_neo = desti_fldr / fn.name
        print("{0}   ---->   {1}".format(fn, fn_neo))
        shutil.move(fn, fn_neo)
# %%
