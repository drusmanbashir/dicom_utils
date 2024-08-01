# %%
# functions to alter dcm headers 
from enum import Enum
import ipdb
tr = ipdb.set_trace
import os

from matplotlib import shutil
from pydicom.fileset import FileSet
from pydicom import dcmread
from pydicom.filereader import read_dicomdir
from pathlib import Path

from dicom_utils.helpers import delete_unwanted_files_folders


vendor = 0x8,0x70
model = 0x0008,0x1090
thickness = 0x18,0x50
kernel=0x18,0x1210
filter_type=0x18,0x1160
ctdi = 0x0018, 0x9345
kvp = 0x18,0x60
current = 0x18,0x1151
exposure_time=0x18,0x9328
exposure=0x18,0x1152 # mAs

def dcm_id_fromfoldername(dcmfldr:Path):
        '''
        dcmfldr: this will become the new patient id. DICOM files will be recursively sought in its subtrees
        '''
        
        new_id = dcmfldr.name
        fldrs = list(dcmfldr.rglob("*"))
        dcm_files =[fldr for fldr in fldrs if fldr.is_file() ]
        for dcm_filename in dcm_files:
            fix_file(dcm_filename,["PatientID"],[new_id])

def rename_tag(dcm_header, tag,value,verbose = True):
    id_element = dcm_header.data_element(tag)
    if verbose==True:
        print("Tag {0} value is changed from {1} to {2} ".format(id_element,id_element.value,value))
    id_element.value=value
    return dcm_header

def fix_file (dcm_filename,tags:list,values:list):
        try:
            hd = dcmread(dcm_filename)
            for tag,value in zip(tags,values):
                hd = rename_tag(hd,tag, value)
                hd.save_as(dcm_filename)
        except:
            print("Not a valid file {}".format(dcm_filename))
# %%
if __name__ == "__main__":
    

    fldrs = Path("/s/xnat_shadow/crc/staging")

    delete_unwanted_files_folders(fldrs)
# %%
    fn = "/s/datasets_bkp/crc_project/DICOM/CRC065/DICOM/0000364C/AA81D01E/AADAEA18/000035A1/EE0ABF76.dcm"
# %%
    tag_list =vendor,model,kernel,filter_type,kvp,exposure,ctdi,thickness
    dc = dcmread(fn)
    print(dc)
    dc[0x0008,0x1090]
    vals = [dc[tag].value for tag in tag_list]
    dc[kernel]
    dc.SliceThickness
    print
    dc[mod.value]
    fldrs = Path()
