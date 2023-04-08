# %%
import math 
import numpy as np
from fran.inference.scoring import ToTensorT
import pydicom_seg
from fastai.data.core import attrdict, inspect
inspect
from pathlib import Path
from pdb import main
from fastai.callback.fp16 import delegates
from fastai.vision.augment import properties, store_attr
import pydicom
from pydicom import dcmread
from pydicom.filereader import read_dicomdir
import shutil
import pandas as pd
from dcm_tags import translate_tag


import SimpleITK as sitk
import sys
import os
from fran.utils.fileio import maybe_makedirs, save_sitk, str_to_path
from fran.utils.helpers import  ask_proceed, int_to_str, multiprocess_multiarg, path_to_str

from fran.utils.imageviewers import ImageMaskViewer, view_sitk
import ipdb

tr = ipdb.set_trace
    
def dcmfolder_to_sitk(series_folder):
            reader = sitk.ImageSeriesReader()
            nms = reader.GetGDCMSeriesFileNames(str(series_folder))
            reader.SetFileNames(nms)
            img = reader.Execute()
            return img


class DCMToSITK:
    def __init__(self, dataset_name, parent, tags:list=None, sitk_ext='.nrrd',  sitk_folder=None, rename_sitk=True,start_ind=0):
        '''
        param parent: should be sub-folder under the dataset folder, ideally named 'dicom'
        params dcm_tags: if chosen, they form the outputname suffix like so '{dataset_name}_{caseid}_{tags}.sitk_ext'
            tags are typically dcm but can also be strings like 'ct' or 't2'
        '''
                
        store_attr(but='start_ind')
        self._ind = start_ind
        self.sitk_folder = sitk_folder
        self.create_cases_summary()

    def create_cases_summary(self): pass

    def create_output_name(self, case_id,  header):
        suffixes = [translate_tag(header,x) for x in self.tags] if self.tags else []
        delim = "_"
        output_filename =  delim.join([ case_id, *suffixes]) + self.sitk_ext
        return self.sitk_folder/output_filename

    def get_dcm_files(self,series_folder):
            self.reader = sitk.ImageSeriesReader()
            nms = self.reader.GetGDCMSeriesFileNames(str(series_folder))
            if len(nms) == 0:
                return None, None
            else: 
                return nms, dcmread(nms[0])
    def maybe_load_dcm(self,nms):
            self.reader.SetFileNames(nms)
            if not self.sitk_folder.exists(): os.makedirs(self.sitk_folder)
            img = self.reader.Execute()
            return img

    def write_sitk(self,dcm_img,output_name):
                print("Saving dicom series as {}".format(output_name))
                writer = sitk.ImageFileWriter()
                writer.SetFileName(str(output_name))
                writer.Execute(dcm_img)

    def dcmseries_to_sitk(
            self, case_id, series_folder, tags:list=None, ext=".nrrd", overwrite=False
    ):
        nms,header = self.get_dcm_files(str(series_folder))
        if nms:
            output_name = self.create_output_name(
                case_id, header
            )
            if overwrite == True or not os.path.exists(output_name):
                dcm_img = self.maybe_load_dcm(nms)
                self.write_sitk(dcm_img,output_name)
            else:
                print(" File {} exists. Skipping..".format(output_name))
        else:   print(" No dicom files in {}".format(series_folder))

    def dcmfolder_to_sitk(self, case_row, overwrite=False):
        series_folder = None
        case_folder = Path(case_row['case_folder'])
        if 'sitk_id' in case_row.keys():
            case_id = case_row['sitk_id']
        else:
            case_id = case_folder.name
        for path in sorted(case_folder.rglob("*")):
            if path.is_file() and path.parent != series_folder:
                series_folder = path.parent
                self.dcmseries_to_sitk(
                    case_id = case_id,
                    series_folder=series_folder,
                    overwrite=overwrite,
                )


    def save_cases_summary(self):
        def _inner():
            self.cases_summary.to_csv(self.cases_summary_fn,index=False)
        if self.cases_summary_fn.exists():
            ask_proceed("File {} already exists. Overwrite?".format(self.cases_summary_fn))(_inner)()
        else:
            _inner()


    @property
    def ind(self): 
        ind= self._ind
        self._ind +=1
        return ind


    @property
    def cases_summary_fn(self):
        return Path(self.parent.parent/("cases_summary.csv"))



    def generate_sitk_id(self):
        id_number = int_to_str(self.ind, 5)
        return "_".join([self.dataset_name ,id_number])

    @property
    def sitk_folder(self):
        return self._sitk_folder

    @sitk_folder.setter
    def sitk_folder(self, value):
        if value:
            self._sitk_folder = value
        else:
            self._sitk_folder = self.parent.parent / ("sitk/images")


def single_row_dcm_to_sitk(cls,dataset_name, parent,tags, rename_sitk,sitk_ext, df_row,overwrite=True):
    D = cls(dataset_name= dataset_name,parent = parent,tags=tags,  rename_sitk = rename_sitk, sitk_ext=sitk_ext)
    D.dcmfolder_to_sitk(df_row)
    return df_row

def delete_unwanted_files_folders(
        parent, delete_these=["SECTRA", "DICOMDIR", "README", "ComponentUpdate", "Viewer"]
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



def mp_cleanup_folders(csv_fn):
    df = pd.read_csv(csv_fn)
    case_folders = [[Path(case_folder)] for case_folder in df['case_folder']]
    res = multiprocess_multiarg(delete_unwanted_files_folders, case_folders,num_processes=6)


def mp_dataset(cls,dataset_name, parent,tags=['StudyData'],rename_sitk=True,sitk_ext='.nrrd',debug=False,overwrite=True):
    D = cls(dataset_name=dataset_name,parent=parent,tags=tags,  rename_sitk=rename_sitk)
    csv_fn = D.cases_summary_fn
    df = pd.read_csv(csv_fn)
    args = [[cls,dataset_name, parent,tags, rename_sitk, sitk_ext, df.iloc[n],overwrite] for n in range(len(df))]
    res = multiprocess_multiarg( single_row_dcm_to_sitk,args,num_processes=8,debug=debug)
    return res

# %%

if __name__ == "__main__":
    parent = Path("/media/ub/datasets/litq_short/dicom")
    D = LITQToSITK("litq", parent,tags= ['StudyDate','StudyDescription'],rename_sitk=True,start_ind=10)
    # D.save_cases_summary()

# %%


    mask_pt = ToTensorT()(mask)
    img_pt = ToTensorT()(img)
    # res = mp_cleanup_folders(D.cases_summary_fn)
    ImageMaskViewer([img_pt,mask_pt])
# %%
    debug = False
    results = mp_dataset(LITQToSITK,'litq',parent,tags =['StudyDate','SliceThickness'],debug=debug)

# %%
    d = D()
# %%

