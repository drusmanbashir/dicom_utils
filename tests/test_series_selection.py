"""Series selection per study: body region by z, kernel class, MAR, phase, thickness, output names."""

import pandas as pd
import pytest
from pydicom.dataset import Dataset

from dicom_utils.dcm_to_sitk import DCMDatasetToSITK
from dicom_utils.series_selection import kernel_class, is_mar, phase
from test_dar043 import series


def convert(tmp_path, max_series=1):
    d = DCMDatasetToSITK('test', tmp_path/'input', tmp_path/'out'/'images', starting_ind=None,
                         tags=['StudyDate', 'SeriesDescription'], max_series_per_case=max_series,
                         min_files_per_series=1, include_modalities=['CT'], exclude_image_type_values=['LOCALIZER'])
    d.create_cases_summary()
    d.save_cases_summary()
    d.process_all_cases(debug=True)
    return sorted(p.name for p in (tmp_path/'out'/'images').glob('*.nii.gz'))


def hdr(desc='', kernel=None):
    h = Dataset()
    h.SeriesDescription = desc
    if kernel:
        h.ConvolutionKernel = kernel
    return h


@pytest.mark.parametrize('desc,kernel,cls', [
    ('CAP 1.5mm', 'I26f\\3', 'soft'), ('Thorax 0.75 I70f 3', 'I70f\\3', 'lung'),
    ('Body Body 1.0 CE', 'BODY_SHARP', 'soft'), ('Lung Lung 1.0 CE', 'LUNG', 'lung'),
    ('3/2 Lung', 'YB', 'lung'), ('Source', 'B', 'soft'), ('_LCAD', 'B70f', 'lung'),
    ('Thorax 1.5 Bl57 3', 'Bl57f\\3', 'lung'), ('Abdomen 3.0 Br38 3', 'Br38f\\3', 'soft'),
])
def test_kernel_class(desc, kernel, cls):
    assert kernel_class(hdr(desc, kernel)) == cls


def test_mar_and_phase_from_description():
    assert is_mar(hdr('O-MAR Source')) and is_mar(hdr('Tho CAP 1.5 Bf37 3 iMAR'))
    assert not is_mar(hdr('Non O-MAR Source'))
    assert phase(hdr('Arterial 1.5')) == 'arterial'
    assert phase(hdr('Non contrast chest')) == 'unenhanced'
    assert phase(hdr('Non O-MAR Source')) == phase(hdr('CAP 5mm')) == 'routine'


def test_two_body_regions_both_converted(tmp_path):
    """Live DAR043 case 5: Thorax over z -129..115 and Abdomen over z -485..-30 are separate scans."""
    case = tmp_path/'input'/'000005'
    series(case/'a', desc='Thorax 0.75', thickness='0.75', z0=-129, span=244)
    series(case/'b', desc='Abdomen 0.75', thickness='0.75', z0=-485, span=455)
    assert convert(tmp_path) == ['test_000005_20250101_abdomen075.nii.gz', 'test_000005_20250101_thorax075.nii.gz']


def test_lung_kernel_inside_cap_is_same_region_but_kept(tmp_path):
    case = tmp_path/'input'/'000009'
    series(case/'a', desc='CAP 1.5mm', thickness='1.5', z0=-720, span=594, kernel='I26f')
    series(case/'b', desc='CAP 5mm', thickness='5', z0=-720, span=594, kernel='I30f')
    series(case/'c', desc='Lung 5mm', thickness='5', z0=-400, span=270, kernel='I70f')
    assert convert(tmp_path) == ['test_000009_20250101_cap15mm.nii.gz', 'test_000009_20250101_lung5mm.nii.gz']


def test_mar_and_non_mar_both_kept(tmp_path):
    case = tmp_path/'input'/'000001'
    series(case/'a', desc='O-MAR Source', thickness='1.5', kernel='B')
    series(case/'b', desc='Non O-MAR Source', thickness='1.5', kernel='B')
    series(case/'c', desc='3/1 recons', thickness='3', kernel='B')
    assert convert(tmp_path) == ['test_000001_20250101_nonomarsource.nii.gz', 'test_000001_20250101_omarsource.nii.gz']


def test_arterial_dropped_only_when_routine_exists(tmp_path):
    series(tmp_path/'input'/'000001'/'a', desc='Arterial 1.5', thickness='1.5')
    series(tmp_path/'input'/'000001'/'b', desc='CAP 1.5mm', thickness='1.5')
    series(tmp_path/'input'/'000002'/'a', desc='Arterial 1.5', thickness='1.5')
    assert convert(tmp_path) == ['test_000001_20250101_cap15mm.nii.gz', 'test_000002_20250101_arterial15.nii.gz']


def test_shared_description_gets_series_number(tmp_path):
    """Live DAR043 cases 18/67/81: 'Body Body 1.0' in two regions."""
    case = tmp_path/'input'/'000018'
    series(case/'a', desc='Body Body 1.0', z0=1407, span=390, number=4)
    series(case/'b', desc='Body Body 1.0', z0=1687, span=355, number=7)
    assert convert(tmp_path) == ['test_000018_20250101_bodybody10_s4.nii.gz', 'test_000018_20250101_bodybody10_s7.nii.gz']
    row = pd.read_csv(tmp_path/'out'/'cases_summary.csv', dtype=str).iloc[0]
    assert row.output == 'test_000018_20250101_bodybody10_s4.nii.gz;test_000018_20250101_bodybody10_s7.nii.gz'


def test_two_study_dates_still_limited(tmp_path):
    series(tmp_path/'input'/'000002'/'a', date='20250101')
    series(tmp_path/'input'/'000002'/'b', date='20250201')
    assert convert(tmp_path) == []
    assert 'study dates' in pd.read_csv(tmp_path/'out'/'cases_summary.csv', dtype=str).iloc[0].reason
