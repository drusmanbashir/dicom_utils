"""Which axial CT series of a study are converted. Rules and their reasons: series_selection.md (repo root).

Per study: one series per (body region, contrast phase, kernel class, metal-artifact reduction),
the thickness variant chosen by pick_by_thickness. Arterial and unenhanced series are dropped
when the same region has a routine (portal-venous) series.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue

REGION_OVERLAP = 0.8  # same region: z-ranges overlap by >= this fraction of the shorter series
MIN_MM = 0.8

LUNG_DESC = re.compile(r"lung|hrct|lcad", re.I)
LUNG_KERNEL = re.compile(r"^(?:LUNG|BONE|Y[A-D]|Bl\d\d|[BI]r?[6-9]\d|Bf[6-9]\d)", re.I)
MAR = re.compile(r"\b(?:o-?mar|imar|mar)\b", re.I)
NON_MAR = re.compile(r"\bnon[\s-]?(?:o-?|i)?mar\b", re.I)
ARTERIAL = re.compile(r"\b(?:arterial|art|angio\w*|cta)\b", re.I)
UNENHANCED = re.compile(r"\b(?:non[\s-]?contrast|non[\s-]?con|unenhanced|plain|pre[\s-]?contrast|nc)\b", re.I)


@dataclass
class Series:
    """One DICOM series folder: first-file header, z extent over all its images, image count."""

    folder: Path
    header: Dataset
    z0: float
    z1: float
    nfiles: int


def _desc(header) -> str:
    return str(getattr(header, "SeriesDescription", ""))


def is_axial(header) -> bool:
    """False if SeriesDescription names cor/sag or the IOP slice normal has |z| <= 0.8.

    Missing or degenerate ImageOrientationPatient counts as axial.
    """
    desc = _desc(header).lower()
    if re.search(r"\bcor\b", desc) or re.search(r"\bsag\b", desc):
        return False
    iop = getattr(header, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        return True
    normal = np.cross(np.array(iop[:3], float), np.array(iop[3:6], float))
    nnorm = float(np.linalg.norm(normal))
    if nnorm == 0:
        return True
    return abs(normal[2]) / nnorm > 0.8


def slice_thickness_mm(header) -> float:
    """SliceThickness in mm; 999 when the tag is missing."""
    value = getattr(header, "SliceThickness", 999)
    if value is None or str(value).strip() == "":
        raise ValueError("Empty SliceThickness")
    thickness = float(value)
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError(f"Invalid SliceThickness: {value}")
    return thickness


def kernel_class(header) -> str:
    """'lung' for lung/sharp reconstructions (by description or kernel code), else 'soft'."""
    kernel = getattr(header, "ConvolutionKernel", "")
    kernel = str(kernel[0] if isinstance(kernel, MultiValue) else kernel).strip()  # 'I70f\\3' reads as 2 values
    if LUNG_DESC.search(_desc(header)) or LUNG_KERNEL.match(kernel):
        return "lung"
    return "soft"


def is_mar(header) -> bool:
    """Metal-artifact-reduced reconstruction (O-MAR, iMAR); 'Non O-MAR' is not."""
    desc = _desc(header)
    return bool(MAR.search(desc)) and not NON_MAR.search(desc)


def phase(header) -> str:
    """'arterial' | 'unenhanced' | 'routine', from SeriesDescription."""
    desc = _desc(header)
    if ARTERIAL.search(desc):
        return "arterial"
    if UNENHANCED.search(desc):
        return "unenhanced"
    return "routine"


def same_region(a: Series, b: Series) -> bool:
    overlap = min(a.z1, b.z1) - max(a.z0, b.z0)
    shorter = min(a.z1 - a.z0, b.z1 - b.z0)
    return overlap >= REGION_OVERLAP * shorter


def region_groups(series: list) -> list:
    """Connected components of same_region, ordered by lowest z."""
    series = sorted(series, key=lambda s: s.z0)
    parent = list(range(len(series)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(series)):
        for j in range(i):
            if same_region(series[i], series[j]):
                parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i, s in enumerate(series):
        groups[root(i)].append(s)
    return list(groups.values())


def pick_by_thickness(series: list, min_mm: float = MIN_MM) -> Series:
    """Any series thinner than min_mm: the thickest of those. Else the thinnest. Ties: more files."""
    below = [s for s in series if slice_thickness_mm(s.header) < min_mm]
    if below:
        return max(below, key=lambda s: (slice_thickness_mm(s.header), s.nfiles))
    return min(series, key=lambda s: (slice_thickness_mm(s.header), -s.nfiles))


def select_series(series: list) -> list:
    """Axial series of one study -> the series to convert, ordered by region then variant."""
    picks = []
    for region in region_groups(series):
        routine = [s for s in region if phase(s.header) == "routine"]
        variants = defaultdict(list)
        for s in routine or region:
            variants[(phase(s.header), kernel_class(s.header), is_mar(s.header))].append(s)
        picks += [pick_by_thickness(variants[key]) for key in sorted(variants)]
    return picks
