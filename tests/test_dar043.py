from pathlib import Path
import logging
import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
from dicom_utils.dcm_to_sitk import DCMDatasetToSITK, read_series
from dicom_utils.series_selection import is_axial as _is_axial, slice_thickness_mm as _slice_thickness_mm


def _series_header(folder):
    return read_series(folder).header


def series(folder, *, date='20250101', thickness='1', desc='axial', uid=None, orientation=None, z0=0, span=1, kernel=None, number=1):
    folder.mkdir(parents=True, exist_ok=True)
    uid = uid or generate_uid()
    study = generate_uid()
    for i in range(2):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f'{uid}.{i}.dcm'), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = uid
        ds.StudyInstanceUID = study
        ds.StudyDate = date
        ds.Modality = 'CT'
        ds.SeriesDescription = desc
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        if thickness is not None:
            ds.SliceThickness = thickness
        ds.ImageOrientationPatient = orientation or [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, z0 + i * span]
        ds.SeriesNumber = number
        if kernel:
            ds.ConvolutionKernel = kernel
        ds.Rows = ds.Columns = 2
        ds.PixelSpacing = [1, 1]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
        ds.save_as(ds.filename, enforce_file_format=True)
    return uid


def converter(tmp_path):
    return DCMDatasetToSITK('test', tmp_path/'input', tmp_path/'out'/'images', starting_ind=None, tags=['StudyDate'], max_series_per_case=1, min_files_per_series=1, include_modalities=['CT'], exclude_image_type_values=['LOCALIZER'])


def test_series_validation_and_orientation(tmp_path):
    folder = tmp_path/'series'
    series(folder, desc='axial corrected')
    assert _is_axial(_series_header(folder))
    series(folder)
    with pytest.raises(ValueError, match='SeriesInstanceUID'):
        _series_header(folder)


def test_missing_and_empty_thickness(tmp_path):
    series(tmp_path/'missing', thickness=None)
    series(tmp_path/'empty', thickness='')
    assert _slice_thickness_mm(_series_header(tmp_path/'missing')) == 999
    with pytest.raises(ValueError):
        _slice_thickness_mm(_series_header(tmp_path/'empty'))


def test_multiprocess_summary_rerun_and_source(tmp_path):
    inp = tmp_path/'input'
    uid = series(inp/'000001'/'a', desc='axial corrected')
    series(inp/'000002'/'a', date='20250101')
    series(inp/'000002'/'b', date='20250201')
    series(inp/'000003'/'a', thickness='')
    series(inp/'000004'/'a', orientation=[1, 0, 0, 0, 0, 1])
    junk = inp/'000001'/'README.txt'
    junk.write_text('keep')
    snapshot = {str(p.relative_to(inp)): (p.stat().st_mtime_ns, p.read_bytes()) for p in inp.rglob('*') if p.is_file()}
    d = converter(tmp_path)
    d.create_cases_summary()
    d.save_cases_summary()
    d.process_all_cases(debug=False)
    assert snapshot == {str(p.relative_to(inp)): (p.stat().st_mtime_ns, p.read_bytes()) for p in inp.rglob('*') if p.is_file()}
    assert junk.exists()
    log = d.logfile.read_text()
    assert "000001: converted" in log
    assert "000003: failed" in log
    summary = pd.read_csv(tmp_path/'out'/'cases_summary.csv', dtype=str).set_index('sitk_id')
    assert summary.loc['test_000001', 'SeriesInstanceUID'] == uid
    assert summary.loc['test_000001', 'SeriesDescription'] == 'axial corrected'
    assert summary.loc['test_000001', 'source_file_count'] == '2'
    assert summary.loc['test_000001', 'SliceThickness'] == '1.0'
    for case in ['000002', '000003', '000004']:
        assert summary.loc['test_'+case, 'reason']
    output = tmp_path/'out'/'images'/'test_000001_20250101.nii.gz'
    before = output.stat().st_mtime_ns
    d = converter(tmp_path)
    d.create_cases_summary()
    d.save_cases_summary()
    d.process_all_cases(debug=False)
    assert output.stat().st_mtime_ns == before
    summary = pd.read_csv(tmp_path/'out'/'cases_summary.csv', dtype=str).set_index('sitk_id')
    assert 'exists' in summary.loc['test_000001', 'reason']
    d.process_all_cases(debug=False, overwrite=True)
    assert output.stat().st_mtime_ns > before


def test_log_without_root_handlers(tmp_path):
    (tmp_path/'input').mkdir()
    root = logging.getLogger()
    old = root.handlers[:]
    for h in old: root.removeHandler(h)
    try:
        d = converter(tmp_path)
        logging.warning('dar043 log probe')
        assert 'dar043 log probe' in d.logfile.read_text()
    finally:
        for h in root.handlers[:]: root.removeHandler(h); h.close()
        for h in old: root.addHandler(h)


def test_legacy_mapping_and_new_cases(tmp_path):
    inp = tmp_path/'input'
    series(inp/'000001'/'a')
    series(inp/'000002'/'a')
    legacy = tmp_path/'cases_summary.csv'
    legacy.write_text(f"case_folder,sitk_id\n{inp/'000001'},test_009999\n")
    before = legacy.read_bytes()
    d = converter(tmp_path)
    d.create_cases_summary()
    d.save_cases_summary()
    d.process_all_cases(debug=True)
    rows = pd.read_csv(tmp_path/'out'/'cases_summary.csv', dtype=str)
    assert list(rows.sitk_id) == ['test_009999', 'test_000002']
    assert list(rows.status) == ['converted', 'converted']
    assert legacy.read_bytes() == before


def test_log_with_existing_root_handler(tmp_path):
    (tmp_path/'input').mkdir()
    root = logging.getLogger()
    existing = logging.NullHandler()
    root.addHandler(existing)
    try:
        d = converter(tmp_path)
        logging.warning('existing handler probe')
        assert existing in root.handlers
        assert 'existing handler probe' in d.logfile.read_text()
    finally:
        root.removeHandler(existing)


def test_mixed_uid_is_rejected(tmp_path):
    folder = tmp_path/'mixed'
    series(folder)
    series(folder)
    with pytest.raises(ValueError, match='SeriesInstanceUID'):
        _series_header(folder)


def test_summary_destination(tmp_path):
    (tmp_path/'input').mkdir()
    d = converter(tmp_path)
    d.create_cases_summary()
    d.save_cases_summary()
    assert (tmp_path/'out'/'cases_summary.csv').is_file()
    assert not (tmp_path/'cases_summary.csv').exists()


def test_failed_write_preserves_existing_output(tmp_path, monkeypatch):
    from dicom_utils.dcm_to_sitk import DCMCaseToSITK
    case = DCMCaseToSITK('test', tmp_path, tmp_path/'out')
    output = tmp_path/'out'/'existing.nii.gz'
    output.write_bytes(b'original')
    class BrokenWriter:
        def SetFileName(self, name):
            self.name = name
        def Execute(self, image):
            Path(self.name).write_bytes(b'partial')
            raise RuntimeError('write failed')
    monkeypatch.setattr('dicom_utils.dcm_to_sitk.sitk.ImageFileWriter', BrokenWriter)
    with pytest.raises(RuntimeError, match='write failed'):
        case.write_sitk(None, output)
    assert output.read_bytes() == b'original'
    assert list(output.parent.iterdir()) == [output]


def test_each_dicom_header_read_once(tmp_path, monkeypatch):
    import dicom_utils.dcm_to_sitk as mod
    series(tmp_path/'input'/'000001'/'a')
    d = converter(tmp_path)
    original = mod.dcmread
    calls = []
    def tracked(path, *args, **kwargs):
        calls.append((str(path), kwargs))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(mod, 'dcmread', tracked)
    result = d.process_single_case(str(tmp_path/'input'/'000001'), 'test_000001')
    assert result['status'] == 'converted'
    assert len(calls) == 2
    assert calls[1][1]['specific_tags'] == ['SeriesInstanceUID', 'ImagePositionPatient']


def test_missing_gdcm_files_is_failure(tmp_path, monkeypatch):
    import dicom_utils.dcm_to_sitk as mod
    series(tmp_path/'input'/'000001'/'a')
    d = converter(tmp_path)
    monkeypatch.setattr(mod.sitk.ImageSeriesReader, 'GetGDCMSeriesFileNames', lambda *args: ())
    result = d.process_single_case(str(tmp_path/'input'/'000001'), 'test_000001')
    assert result['status'] == 'failed'
    assert 'dicom files' in result['reason'].lower()


def test_mhd_sidecar_is_written_next_to_output(tmp_path):
    import SimpleITK as sitk
    from dicom_utils.dcm_to_sitk import DCMCaseToSITK
    case = DCMCaseToSITK('test', tmp_path, tmp_path/'out')
    output = tmp_path/'out'/'image.mhd'
    case.write_sitk(sitk.Image([2, 2, 2], sitk.sitkUInt16), output)
    assert (output.parent/'image.raw').is_file()
    assert sitk.ReadImage(str(output)).GetSize() == (2, 2, 2)
