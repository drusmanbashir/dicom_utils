# dicom_utils

## Core reusable API

- `dicom_utils.rtstruct.RTStructStudy`
  - Loads a mixed DICOM study folder containing CT + RTSTRUCT.
  - Exposes ROI names via `available_rois()`.
  - Generates voxel masks for an ROI via `mask_for_roi(...)`, aligned to the CT grid.

### Minimal usage

```python
from dicom_utils import RTStructStudy

study = RTStructStudy.from_study_dir("/path/to/mixed_dicom_study")
print(study.available_rois())
mask = study.mask_for_roi(study.available_rois()[0], label_value=1)
```

