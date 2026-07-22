from pathlib import Path
import os

import multiprocessing as mp
import pylidc as pl
from utilz.fileio import maybe_makedirs

from lidc import LIDCXNATProcessor


def _default_worker_count(target_utilization=0.70):
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except Exception:
        cpu_count = os.cpu_count() or 1
    return max(1, int(round(cpu_count * target_utilization)))


def _process_scan_idx(
    indx,
    processor_cls,
    clevel=.35,
    imgs_fldr="/s/tmp/images",
    lms_fldr="/s/tmp/masks",
    upload_on_xnat=True,
):
    L = processor_cls(imgs_fldr=imgs_fldr, lms_fldr=lms_fldr)
    L.process_scan(L.scans[indx], clevel=clevel, upload_on_xnat=upload_on_xnat)


class LIDCLocalProcessor(LIDCXNATProcessor):
    """LIDC processor variant that only writes image/LM files locally."""

    def __init__(self, imgs_fldr="/s/tmp/images", lms_fldr="/s/tmp/masks") -> None:
        if pl is None:
            raise ImportError("pylidc is required for LocalLIDCProcessor")
        self.proj_title = "lidc"
        self.proj = None
        self.scans = pl.query(pl.Scan).filter()
        print(self.scans.count())
        self.imgs_fldr = Path(imgs_fldr)
        self.lm_fldr = Path(lms_fldr)
        self._scan_index_map = None
        maybe_makedirs([self.imgs_fldr, self.lm_fldr])

    def process_scan(self, scan, clevel=.35, upload_on_xnat=False):
        # Force local-only behavior.
        super().process_scan(scan, clevel=clevel, upload_on_xnat=False)

    def maybe_upload_scan_rscs(self, scan, fn_img, fn_lm):
        # Local-only processor: no XNAT upload.
        return None

    def maybe_upload_rsc(self, exp, label, fpath):
        # Local-only processor: no XNAT upload.
        return None


def write_dataset_index_csv(processor, dataset_root, csv_path=None):
    """Convenience wrapper to emit filename/index CSV for a dataset folder."""
    ds = Path(dataset_root)
    images = ds / "images"
    lms = ds / "lms"
    if not images.exists() or not lms.exists():
        raise FileNotFoundError("Dataset folder must contain images/ and lms/: {0}".format(ds))
    L = processor(imgs_fldr=images, lms_fldr=lms)
    return L.write_index_csv(dataset_root=ds, csv_path=csv_path)


def _build_args_from_selection(
    L,
    processor,
    candidate_indices,
    imgs_fldr,
    lms_fldr,
    clevel,
    upload_on_xnat,
    overwrite,
    existing_indices,
    existing_filenames,
):
    args = []
    seen_filenames = set(existing_filenames)
    for indx in candidate_indices:
        fn = L.get_filename_from_index(indx)
        if not overwrite:
            if indx in existing_indices:
                continue
            if fn in seen_filenames:
                continue

        fn_img = Path(imgs_fldr) / fn
        fn_lm = Path(lms_fldr) / fn
        if overwrite or not (fn_img.exists() and fn_lm.exists()):
            args.append((indx, processor, clevel, imgs_fldr, lms_fldr, upload_on_xnat))
            seen_filenames.add(fn)
    return args


def lidc_multiproc(
    processor,
    imgs_fldr,
    lms_fldr,
    indices=None,
    num_processes=None,
    clevel=.35,
    overwrite=False,
    upload_on_xnat=False,
    complete_n_cases=False,
    n_cases=None,
    existing_index_csvs=None,
):

    assert processor in [LIDCLocalProcessor, LIDCXNATProcessor], "processor must be LIDCLocalProcessor or LIDCXNATProcessor"
    L = processor(imgs_fldr=imgs_fldr, lms_fldr=lms_fldr)
    total_scans = L.scans.count()

    # Default selection behavior.
    if indices is None:
        candidate_indices = list(range(total_scans))
    elif isinstance(indices, int):
        candidate_indices = list(range(min(indices, total_scans)))
    else:
        candidate_indices = [int(indx) for indx in indices]

    existing_indices = set()
    existing_filenames = set()
    if not overwrite and existing_index_csvs:
        existing_indices, existing_filenames = L.gather_existing_entries_from_csvs(existing_index_csvs)
        print(
            "Found {0} existing indexes and {1} existing filenames from CSVs".format(
                len(existing_indices), len(existing_filenames)
            )
        )
        print("CSV sources: {0}".format([str(Path(p)) for p in existing_index_csvs]))

    if num_processes is None or num_processes < 1:
        num_processes = _default_worker_count(0.70)

    if overwrite:
        args = _build_args_from_selection(
            L=L,
            processor=processor,
            candidate_indices=candidate_indices,
            imgs_fldr=imgs_fldr,
            lms_fldr=lms_fldr,
            clevel=clevel,
            upload_on_xnat=upload_on_xnat,
            overwrite=True,
            existing_indices=set(),
            existing_filenames=set(),
        )
    else:
        if n_cases is None and isinstance(indices, int):
            n_cases = indices

        if complete_n_cases and n_cases is not None:
            if n_cases <= 0:
                print("No cases requested (n_cases <= 0)")
                return []
            args = []
            seen_filenames = set(existing_filenames)
            for indx in range(total_scans):
                fn = L.get_filename_from_index(indx)
                if indx in existing_indices:
                    continue
                if fn in seen_filenames:
                    continue
                fn_img = Path(imgs_fldr) / fn
                fn_lm = Path(lms_fldr) / fn
                if fn_img.exists() and fn_lm.exists():
                    seen_filenames.add(fn)
                    continue
                args.append((indx, processor, clevel, imgs_fldr, lms_fldr, upload_on_xnat))
                seen_filenames.add(fn)
                if len(args) >= n_cases:
                    break
            if len(args) < n_cases:
                print(
                    "Requested {0} fresh cases but only queued {1} (exhausted scan list after skips).".format(
                        n_cases, len(args)
                    )
                )
        else:
            args = _build_args_from_selection(
                L=L,
                processor=processor,
                candidate_indices=candidate_indices,
                imgs_fldr=imgs_fldr,
                lms_fldr=lms_fldr,
                clevel=clevel,
                upload_on_xnat=upload_on_xnat,
                overwrite=False,
                existing_indices=existing_indices,
                existing_filenames=existing_filenames,
            )

    queued_indices = [a[0] for a in args]
    print("Queued {0} scans".format(len(args)))
    if queued_indices:
        print("First queued indexes: {0}".format(queued_indices[:20]))

    if not args:
        return []

    print("Using {0} worker processes".format(num_processes))
    try:
        # On Linux, using fork avoids pickling __main__ callables under spawn.
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    with ctx.Pool(processes=num_processes) as pool:
        pool.starmap(_process_scan_idx, args)

    return queued_indices


# %%
if __name__ == "__main__":
    # Example: complete to 200 fresh cases by skipping indexes present in provided CSVs.
    lidc_multiproc(
        processor=LIDCLocalProcessor,
        imgs_fldr="/media/UB/datasets/lidc2/images",
        lms_fldr="/media/UB/datasets/lidc2/lms",
        indices=200,
        overwrite=False,
        complete_n_cases=True,
        existing_index_csvs=[
            "/media/UB/datasets/lidc/lidc_indices.csv",
            "/media/UB/datasets/lidc2/lidc2_indices.csv",
        ],
    )

# %%
