# %%
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
import numpy as np
from pathlib import Path
import datetime

from xnat.helpers import maybe_makedirs
import os
def nifti_rgb_to_dicom_series(
    nifti_path,
    ref_dicom_dir,
    output_dir,
    series_description="RGB Overlay Series",
    study_instance_uid=None,
    series_instance_uid=None,
    signed=True,
):
    """
    Convert a 4D NIfTI (nx, ny, nz, 3) to a DICOM series of 2D color images,
    using essential metadata from a reference DICOM series.

    :param nifti_path: Path to the input NIfTI file (last dim = 3 for RGB).
    :param ref_dicom_dir: Directory containing the reference DICOM series.
    :param output_dir: Output directory for the new DICOM files.
    :param series_description: DICOM SeriesDescription for the new series.
    :param study_instance_uid: (Optional) If provided, overrides the study UID.
    :param series_instance_uid: (Optional) If provided, overrides the series UID.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----------------------------
    # 1) Load reference DICOM
    # ----------------------------
    ref_dicom_files = [
        f for f in os.listdir(ref_dicom_dir)
        if os.path.isfile(os.path.join(ref_dicom_dir, f))
    ]
    if len(ref_dicom_files) == 0:
        raise ValueError(f"No DICOM files found in {ref_dicom_dir}")

    # Just read the first file as reference (for geometry, spacing, etc.)
    ref_dicom_path = os.path.join(ref_dicom_dir, ref_dicom_files[0])
    ref_ds = pydicom.dcmread(ref_dicom_path)

    # ----------------------------
    # 2) Load the RGB NIfTI
    # ----------------------------

    image = sitk.ReadImage(nifti_path)
    origin = np.array(image.GetOrigin())  # (x0, y0, z0)
    direction = np.array(image.GetDirection()).reshape(3, 3)
    nifti_data2 = sitk.GetArrayFromImage(image)
    dimension = image.GetDimension()
    num_components = image.GetNumberOfComponentsPerPixel()
    size_x, size_y, size_z = image.GetSize()

    # The SITK spacing is (spacingX, spacingY, spacingZ)
    spacing_x, spacing_y, spacing_z = image.GetSpacing()
    # nifti_data2 = np.swapaxes(nifti_data2, 0, 2)
    # Convert to uint8 or uint16 as needed. Let's assume 8-bit overlay:

    # Check shape
    if len(nifti_data2.shape) != 4 or nifti_data2.shape[3] != 3:
        raise ValueError(
            "NIfTI must be 4D with last dim=3 for RGB, but got shape "
            f"{nifti_data2.shape}"
        )


    # In DICOM, typically the array shape is (rows, columns) = (ny, nx)
    # We'll keep the same orientation as reference, so DICOM sees (Rows=ny, Columns=nx).
    # We'll create one 2D color image per slice in z.

    # ----------------------------
    # 3) Prepare new UIDs
    # ----------------------------
    if study_instance_uid is None:
        # Optionally reuse reference or create new
        study_instance_uid = ref_ds.StudyInstanceUID
    if series_instance_uid is None:
        series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    # ----------------------------
    # 4) Loop through slices in z
    # ----------------------------
    pixel_rep = 1

    for z in range(size_z):
        # Extract the 2D slice (still 3 components per pixel)
        # slice_img = image[:, :, z]  # shape in SITK terms: (sizeX, sizeY), 3 components
        # Convert to numpy array => shape (height, width, 3) => (sizeY, sizeX, 3)
        # slice_array = sitk.GetArrayFromImage(slice_img)
        slice_array = nifti_data2[z,:,:,:]
        # Typically this is (sizeY, sizeX, 3) for a 2D + vector image in SITK.

        # If your original data is float or a higher bit depth, you can scale/clip as needed.
        if signed:
            slice_array = slice_array.astype(np.int16)   # 16-bit signed
            pixel_rep = 1
        else:
            slice_array = slice_array.astype(np.uint16)  # 16-bit unsigned
            pixel_rep = 0

        # Flatten to bytes in row-major order
        pixel_bytes = slice_array.tobytes()
        slice_normal = direction[:, 2]
        ipp = origin + z * spacing_z * slice_normal
        image_position_patient = [x for x in ipp]
        # ----------------------------
        # 4) Build minimal DICOM dataset
        # ----------------------------
        ds = pydicom.Dataset()
        file_meta = pydicom.Dataset()

        # File meta info
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # UIDs
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid

        # Basic patient/study/series
        ds.Modality = "OT"  # or "SC" (Secondary Capture)
        ds.SeriesDescription = series_description
        ds.PatientName = ref_ds.PatientName
        ds.PatientID = ref_ds.PatientID

        ds.SeriesNumber = 1
        ds.InstanceNumber = z + 1

        # Color image attributes (16-bit)
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0  # RGBRGB...
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = pixel_rep  # 0=unsigned, 1=signed

        ds.Rows = size_y
        ds.Columns = size_x

        # ----------------------------
        # 5) Embed SITK-based spacing
        # ----------------------------
        # In DICOM, PixelSpacing is a 2-element list: [row spacing, column spacing].
        ds.PixelSpacing = ref_ds.PixelSpacing
        ds.SliceThickness = ref_ds.SliceThickness
        ds.ImageOrientationPatient = ref_ds.ImageOrientationPatient
        ds.ImagePositionPatient = image_position_patient
        # If you want partial orientation info or position, you could do:
        # ds.ImageOrientationPatient = [ "1", "0", "0", "0", "1", "0" ] # Identity
        # ds.ImagePositionPatient = [ "0", "0", f"{z*spacing_z:.4f}" ]  # if purely axial
        # But for fully correct 3D orientation, parse image.GetDirection().

        ds.PixelData = pixel_bytes

        # ----------------------------
        # 6) Save the DICOM slice
        # ----------------------------
        out_filename = f"slice_{z+1:03d}.dcm"
        out_path = os.path.join(output_dir, out_filename)
        ds.save_as(out_path, write_like_original=False)



    print(f"Color DICOM series written to: {output_dir}")

# %%
if __name__ == "__main__":
    nifti_path = Path("/s/fran_storage/predictions/litsmc/LITS-933/4.nii.gz")
    dcm_folder = Path("/home/ub/Desktop/10_CT_1/4/DICOM")
    output_folder = Path("/home/ub/Desktop/10_CT_1/4/SEG")
    maybe_makedirs (output_folder)
    imgs_tmp = ["/home/ub/Desktop/10_CT_1/nifti/4.nii.gz"]
    dcm_fldr = Path("/home/ub/Desktop/10_CT_1/4/DICOM")
    # preds = En.run(imgs_tmp,chunksize=1)
    nifti_file = "/home/ub/Desktop/10_CT_1/4/overlay/4.nii.gz"
    output_directory = "/home/ub/Desktop/10_CT_1/4/DICOM_OVERLAY"


    nifti_rgb_to_dicom_series(
        nifti_path=nifti_file,
        ref_dicom_dir=dcm_fldr,
        output_dir=output_directory,
        series_description="My RGB Series",
    )
# %%


# %%
