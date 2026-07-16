from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon


@dataclass
class RTStructStudy:
    study_dir: Path
    rtstruct_path: Path
    ct_image: sitk.Image
    roi_number_to_name: dict[int, str]
    roi_name_to_number: dict[str, int]
    roi_contours: dict[int, list]

    @classmethod
    def from_study_dir(cls, study_dir: str | Path) -> "RTStructStudy":
        study_dir = Path(study_dir)
        dcm_files = sorted(study_dir.rglob("*.dcm"))
        if not dcm_files:
            raise RuntimeError(f"No DICOM files found in {study_dir}")

        rtstruct_path = None
        ct_files: list[Path] = []
        ct_meta = []
        for fp in dcm_files:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            mod = str(getattr(ds, "Modality", ""))
            if mod == "RTSTRUCT":
                rtstruct_path = fp
            elif mod == "CT":
                ct_files.append(fp)
                inst = int(getattr(ds, "InstanceNumber", 0) or 0)
                z = None
                if hasattr(ds, "ImagePositionPatient"):
                    z = float(ds.ImagePositionPatient[2])
                ct_meta.append((fp, inst, z))

        if rtstruct_path is None:
            raise RuntimeError(f"No RTSTRUCT file found in {study_dir}")
        if not ct_files:
            raise RuntimeError(f"No CT files found in {study_dir}")

        if all(z is not None for _, _, z in ct_meta):
            ct_meta = sorted(ct_meta, key=lambda x: x[2])
        else:
            ct_meta = sorted(ct_meta, key=lambda x: x[1])
        ordered_ct_files = [str(x[0]) for x in ct_meta]

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(ordered_ct_files)
        ct_image = reader.Execute()

        rt = pydicom.dcmread(str(rtstruct_path), force=True)
        roi_number_to_name = {
            int(x.ROINumber): str(x.ROIName)
            for x in getattr(rt, "StructureSetROISequence", [])
        }
        roi_name_to_number = {v: k for k, v in roi_number_to_name.items()}
        roi_contours = {}
        for item in getattr(rt, "ROIContourSequence", []):
            roi_num = int(item.ReferencedROINumber)
            contours = list(getattr(item, "ContourSequence", []))
            roi_contours[roi_num] = contours

        return cls(
            study_dir=study_dir,
            rtstruct_path=rtstruct_path,
            ct_image=ct_image,
            roi_number_to_name=roi_number_to_name,
            roi_name_to_number=roi_name_to_number,
            roi_contours=roi_contours,
        )

    def available_rois(self) -> list[str]:
        return [self.roi_number_to_name[k] for k in sorted(self.roi_number_to_name)]

    def mask_for_roi(self, roi: int | str, label_value: int = 1) -> sitk.Image:
        if isinstance(roi, str):
            roi_num = self.roi_name_to_number[roi]
        else:
            roi_num = int(roi)
        contours = self.roi_contours[roi_num]
        arr = np.zeros(self.ct_image.GetSize()[::-1], dtype=np.uint16)  # z, y, x
        sx, sy, sz = self.ct_image.GetSize()

        for contour in contours:
            data = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            idx = np.asarray(
                [self.ct_image.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in data],
                dtype=float,
            )
            if idx.size == 0:
                continue
            k = int(round(float(np.mean(idx[:, 2]))))
            if k < 0 or k >= sz:
                continue
            xs = idx[:, 0]
            ys = idx[:, 1]
            rr, cc = polygon(ys, xs, shape=(sy, sx))
            arr[k, rr, cc] = label_value

        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(self.ct_image)
        return out
