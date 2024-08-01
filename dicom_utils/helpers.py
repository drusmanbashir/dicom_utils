
# %%
import numpy as np
import shutil
from pathlib import Path
import SimpleITK as sitk
import re
from pydicom import dcmread
import pydicom_seg
def folder_to_case_id(folder):
    name = folder.name
    pat = r".*\-(\d+)"
    id = re.match(pat,name ,re.IGNORECASE)
    return id.groups(0)[0]

def set_metadata_as(clean:sitk.Image, img:sitk.Image)->sitk.Image:
    clean.SetDirection(img.GetDirection())
    clean.SetOrigin(img.GetOrigin())
    clean.SetSpacing(img.GetSpacing())
    return clean

def get_diff_z(img,mask,tol=1e-3 ):
    spacing = img.GetSpacing()
    im_z_start = img.GetOrigin()[-1]
    z_spacing = spacing[-1]
    mask_z_start = mask.GetOrigin()[-1]
    diff_z = (mask_z_start-im_z_start)/(z_spacing)
    round_up = (1-diff_z)%1
    round_down = diff_z%1
    if round_up<tol: func = np.ceil
    elif round_down<tol: func = np.floor
    else:
        raise "z difference is not in whole numbers! {}".format(diff_z)
    return int(func(diff_z))

def dcm_segmentation(mask_fn):
    dcm_seg = dcmread(mask_fn)
    reader = pydicom_seg.MultiClassReader()
    result = reader.read(dcm_seg)
    mask = result.image
    return mask



from fran.utils.string import int_to_str

def delete_unwanted_files_folders(
        parent, delete_these=["SECTRA",  "README", "ComponentUpdate", "Viewer","DICOMDIR"]
    ):
        dd = list(parent.rglob("*"))
        for dirr in dd:
            if dirr.exists():
                if any((match := substring) in str(dirr) for substring in delete_these):
                    print("Deleting {}".format(dirr))
                    if dirr.is_file() == True:
                        dirr.unlink()
                    else:
                        shutil.rmtree(dirr)

def process_attr(func):
    def _inner(obj,val):
        res = getattr(obj,val)
        return func(val,res)
    return _inner

def kate_style(fldr):
    fldrs = list(fldr.glob("*"))
    for fl in fldrs:
        nm = fl.name
        if nm.isnumeric():
            nm_neo = int_to_str(nm,3)
            nm_neo = "CRC"+nm_neo
            nm_out = fl.parent/nm_neo
            shutil.move(fl,nm_out)
                        
# %%

if __name__ == "__main__":
     
    fldr = Path("/s/datasets_bkp/crc_project/stagingarea/")
    kate_style(fldr)
# %%
