# %%
import itertools as il
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path

import ipdb
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from dicom_utils.dcm_tags import translate_tag
from fastcore.basics import Union, store_attr
from dicom_utils.helpers import delete_unwanted_files_folders
from pydicom import dcmread

from utilz.fileio import maybe_makedirs, save_sitk, str_to_path
from utilz.helpers import ask_proceed, multiprocess_multiarg
from utilz.imageviewers import ImageMaskViewer, view_sitk
from utilz.stringz import int_to_str

tr = ipdb.set_trace

_exclude = {
    "ImageType": "DERIVED",
}


def int_from_string(s):
    a = re.search(r"\d+", s)
    b = int(a.group())
    return b


str_to_path(0)


def non_empty_subfolders(folder, min_files=1):
    fldrs = list(folder.rglob("*"))
    ser_f = [fldr.parent for fldr in fldrs if fldr.is_file()]
    ser_f = set(ser_f)
    ser_f = ser_f.difference([folder])
    final = []
    for ser in ser_f:
        nfiles = len(os.listdir(ser))
        if nfiles < min_files:
            logging.info(
                "Folder {0} has {1} files and is being skipped. Decrease min_files_per_series threshold to include".format(
                    ser, nfiles
                )
            )
        else:
            final.append(ser)
    return final

    folders = []
    excluded_folders = []
    for x in folder.rglob("*"):
        if x.is_file() and x.parent not in il.chain(folders, excluded_folders):
            candidate_fldr = x.parent
            nfiles = len(list(candidate_fldr.glob("*")))
            if nfiles >= min_files:
                folders.append(candidate_fldr)
            else:
                excluded_folders.append(candidate_fldr)
                logging.info(
                    "Folder {0} has {1} files and is being skipped. Decrease min_files_per_series threshold to include".format(
                        candidate_fldr, nfiles
                    )
                )
    return folders


class DCMDatasetToSITK:
    """
    wrapper class processes all dicom cases inside a folder into sitk files. Maintains records in a csv file. If cases already present, it skips (unless overwrite)
    folder: Contains subfolders, each of which is a unique case

    """

    def __init__(
        self,
        dataset_name,
        input_folder,
        output_folder=None,
        starting_ind: Union[None, int] = 0,
        tags: list = None,
        max_series_per_case=3,
        min_files_per_series=50,
        sitk_ext=".nii.gz",
        include_modalities: Union[None, list] = None,
        exclude_image_type_values: Union[None, list] = None,
    ):
        """
        if starting_ind=None, no names are generated. Instead folder names are used as unique ids
        """
        store_attr()
        self.rename_sitk = True if isinstance(starting_ind, int) else False
        self.cases = [case for case in self.input_folder.glob("*") if case.is_dir()]
        self.colnames = ["case_folder", "sitk_id"]
        self.logfile = self.output_folder.parent / "log.txt"
        if not self.output_folder.exists():
            os.makedirs(self.output_folder)
        if self.cases_summary_fn.exists():
            print("A Summary exists. Loading")
            self.cases_summary = pd.read_csv(self.cases_summary_fn)
        else:
            print(
                "A new dataset is initialized in folder {}".format(self.output_folder)
            )
            self.cases_summary = pd.DataFrame(columns=self.colnames)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            filemode = "a" if not self.logfile.exists() else "w"
            logging.basicConfig(
                level=logging.INFO,
                filename=self.logfile,
                encoding="utf-8",
                filemode=filemode,
            )

    def generate_sitk_id(self, ind: int):
        id_number = int_to_str(ind, 5)
        return "_".join([self.dataset_name, id_number])

    def fix_cases_summary(self):
        """
        verifies cases exist in their folders. Removes from csv those cases whose files have been deleted.
        """

        pass

    @property
    def new_cases(self):
        if len(self.cases_summary) == 0:
            new_cases = self.cases
        else:
            cases_in_csv = list(self.cases_summary.case_folder)
            existing = map(Path, cases_in_csv)
            new_cases = set(self.cases).difference(set(existing))
            return new_cases
        return new_cases

    def create_cases_summary(self):
        """
        append: does not overwrite existing id's when new cases are added. Ensures data integrity
        """

        if len(self.new_cases) == 0:
            print(
                "No new cases have been added. the cases_summary on file is not changed"
            )

        else:
            print(
                "Summary file {} exists. Set append to False if you want a fresh start".format(
                    self.cases_summary_fn
                )
            )
            if self.rename_sitk == True:
                indices = self.generate_indices(fill_gaps=True)
            else:
                indices = [fn.name for fn in self.cases]
            cases_summary = [
                [str(case), self.generate_sitk_id(ind)]
                for case, ind in zip(self.new_cases, indices)
            ]
            dftmp = pd.DataFrame(cases_summary, columns=self.colnames)
            self.cases_summary = pd.concat([self.cases_summary, dftmp])
        print(self.cases_summary)

    def save_cases_summary(self):
        def _inner():
            self.cases_summary.to_csv(self.cases_summary_fn, index=False)

        if self.cases_summary_fn.exists():
            ask_proceed(
                "File {} already exists. Overwrite?".format(self.cases_summary_fn)
            )(_inner)()
        else:
            _inner()

    def generate_indices(self, fill_gaps=True):
        taken_ids = list(self.cases_summary.sitk_id)
        if len(taken_ids) > 0:
            bb = list(map(int_from_string, taken_ids))
            largest_ind = max(bb)
            ref = list(range(largest_ind))
            gaps = list(set(ref).difference(bb)) if fill_gaps == True else []
            num_new_inds = len(self.new_cases) - len(gaps)
            starting_ind = largest_ind + 1
            new_inds = gaps + list(range(starting_ind, starting_ind + num_new_inds))
        else:
            new_inds = list(
                range(self.starting_ind, self.starting_ind + len(self.cases))
            )
        return new_inds

    def process_all_cases(self, sitk_ext=".nii.gz", debug=False, overwrite=False):
        args = [
            [row.case_folder, row.sitk_id, sitk_ext, overwrite]
            for i, row in self.cases_summary.iterrows()
        ]
        res = multiprocess_multiarg(self.process_single_case, args, 16, True, debug)
        print("Check {} for errors".format(self.logfile))

    def process_single_case(
        self, dcm_folder, sitk_id, sitk_ext=".nii.gz", overwrite=True
    ):

        D = DCMCaseToSITK(
            self.dataset_name,
            dcm_folder,
            self.output_folder,
            case_id=sitk_id,
            tags=self.tags,
            max_series_per_case=self.max_series_per_case,
            min_files_per_series=self.min_files_per_series,
            sitk_ext=".nii.gz",
            include_modalities=self.include_modalities,
            exclude_image_type_values=self.exclude_image_type_values,
        )
        D.process(overwrite=overwrite)

    @property
    def cases_summary_fn(self):
        return Path(self.input_folder.parent / ("cases_summary.csv"))

    @property
    def input_folder(self):
        return self._input_folder

    @input_folder.setter
    def input_folder(self, value):
        self._input_folder = Path(value)

    @property
    def output_folder(self):
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value):
        if value:
            self._output_folder = Path(value)
        else:
            self._output_folder = self.input_folder/ ("sitk/images")


class DCMCaseToSITK:
    def __init__(
        self,
        dataset_name,
        case_folder,
        output_folder,
        case_id=None,
        tags: list = None,
        max_series_per_case=2,
        min_files_per_series=1,
        sitk_ext=".nii.gz",
        include_modalities: Union[None, list] = None,
        exclude_image_type_values: Union[None, list] = None,
    ):
        """
        converts a single folder with DICOM  files into sitk files. One sitk per DCM series
        parent: should be sub-folder under the dataset folder, ideally named 'dicom'
        dcm_tags: if chosen, they form the outputname suffix like so '{dataset_name}_{caseid}_{tags}.sitk_ext'. In their absence, separate series will have identical sitk names and will overwrite eachother!
            tags are typically dcm but can also be strings like 'ct' or 't2'
        """
        case_folder, output_folder = [Path(f) for f in [case_folder, output_folder]]
        if not case_id:
            case_id = case_folder.name
        store_attr()
        # Preserve old behavior unless caller opts out.
        if self.exclude_image_type_values is None:
            self.exclude_image_type_values = list(_exclude.values())
        if not self.output_folder.exists():
            os.makedirs(self.output_folder)

    def sitk_name_from_series(self, header):
        suffixes = [translate_tag(header, x) for x in self.tags] if self.tags else []
        delim = "_"
        output_filename = delim.join([self.case_id, *suffixes]) + self.sitk_ext
        return self.output_folder / output_filename

    def get_dcm_files(self, series_folder):
        self.reader = sitk.ImageSeriesReader()
        nms = self.reader.GetGDCMSeriesFileNames(str(series_folder))
        if len(nms) == 0:
            return None, None
        else:
            return nms, dcmread(nms[0])

    def maybe_load_dcm(self, nms):
        self.reader.SetFileNames(nms)
        img = self.reader.Execute()
        return img

    def write_sitk(self, dcm_img, output_name):
        print("Saving dicom series as {}".format(output_name))
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(output_name))
        writer.Execute(dcm_img)

    def dcmseries_to_sitk(self, series_folder, overwrite=False):
        """
        processes series, and creates corresponding sitk filename on the fly
        """

        nms, header = self.get_dcm_files(str(series_folder))
        if nms:
            output_name = self.sitk_name_from_series(header)
            if overwrite == True or not os.path.exists(output_name):
                dcm_img = self.maybe_load_dcm(nms)
                self.write_sitk(dcm_img, output_name)
            else:
                logging.info(" File {} exists. Skipping..".format(output_name))
        else:
            logging.warning(" No dicom files in {}".format(series_folder))
            output_name = None
        return output_name

    def process(self, overwrite=False):
        series_folders = self.exclude_unsuitable()
        num_seris = len(series_folders)
        logging.info(
            "Number of series in case_folder {0} is {1}".format(
                self.case_folder, num_seris
            )
        )
        if num_seris > self.max_series_per_case:
            logging.warning(
                "Number of series in case_folder exceeds {}. Please manually remove extra series and retry".format(
                    self.max_series_per_case
                )
            )
        else:
            self.process_all_series(series_folders, overwrite)

    def process_all_series(self, series_folders, overwrite):
        self.output_names=[]
        for folder in series_folders:
            print("Processing folder {}".format(folder))
            output_name = self.dcmseries_to_sitk(
                series_folder=folder,
                overwrite=overwrite,
            )
            print(".. Done.".format(folder))
            self.output_names.append(output_name)
        print("Check attribute output_names for filenames.")

    def exclude_unsuitable(self):
        """
        Parses subfolders and removes those which meet exclusion criteria
        """
        delete_unwanted_files_folders(self.case_folder)
        folders = non_empty_subfolders(
            self.case_folder, min_files=self.min_files_per_series
        )
        folders_suitable = il.filterfalse(self.meets_exclusion_criteria, folders)

        folders_suitable = [s for s in folders_suitable]
        return folders_suitable

    def meets_exclusion_criteria(self, dcm_fldr):
        """
        dcm_fldr has dicom files. Tag of first file is checked against _exclude dict


        """
        fns = list(dcm_fldr.glob("*"))
        he = dcmread(fns[0])
        if self.include_modalities:
            keep_mods = {str(m).upper() for m in self.include_modalities}
            modality = str(getattr(he, "Modality", "")).upper()
            if modality not in keep_mods:
                return True

        if self.exclude_image_type_values:
            image_type = [str(x).upper() for x in getattr(he, "ImageType", [])]
            for val in self.exclude_image_type_values:
                val_up = str(val).upper()
                if any(val_up in im for im in image_type):
                    return True
        return False


def mp_cleanup_folders(folder):
    case_folders = [case_folder for case_folder in folder.glob("*")]
    res = multiprocess_multiarg(
        delete_unwanted_files_folders, case_folders, num_processes=6
    )


class LITQToSITK(DCMDatasetToSITK):
    def __init__(self, dataset_name, input_folder, output_folder=None, *args, **kwargs):
        super().__init__(
            dataset_name,
            input_folder,
            output_folder=output_folder,
            tags=["StudyDate"],
            *args,
            **kwargs
        )


# %%

if __name__ == "__main__":
    # D = LITQToSITK(dataset_name='litq', input_folder='/s/datasets_bkp/litq/dicom')
    D = DCMDatasetToSITK(
        dataset_name="react",
        input_folder="/s/insync/react/",
        max_series_per_case=2,
        tags=["StudyDate"],
        starting_ind=None,
    )
# %%
#SECTION:-------------------- TCIA-CRLM--------------------------------------------------------------------------------------
# %%

    D = DCMDatasetToSITK(
        dataset_name="tciacrlm",
        input_folder="/home/ub/Downloads/dd/CRLM-CT-1009_CT_1",
        max_series_per_case=2,
        tags=["StudyDate"],
        starting_ind=None,
    )
# %%
    D.create_cases_summary()
    D.save_cases_summary()
    D.process_all_cases(debug=False, overwrite=True)
# %%
    fn = "/s/datasets_bkp/manifest-1680277513580/CT Lymph Nodes/ABD_LYMPH_001/09-14-2014-ABDLYMPH001-abdominallymphnodes-30274/300.000000-Lymph node segmentations-27473/1-1.dcm"
    fn = Path(fn)
    pr = fn.parent
    a = dcm_segmentation(fn

    )
    fnseg = pr/("seg.nii.gz")
    sitk.WriteImage(a,fnseg)

# %%
    case_folder = Path("/media/ub/datasets/litq_short/dicom")
    D = LITQToSITK(
        "litq",
        case_folder,
        tags=["StudyDate", "StudyDescription"],
        rename_sitk=True,
        start_ind=10,
    )

# %%

    mask_pt = ToTensorT()(mask)
    img_pt = ToTensorT()(img)
    # res = mp_cleanup_folders(D.cases_summary_fn)
    ImageMaskViewer([img_pt, mask_pt])
# %%
    debug = False
    results = mp_dataset(
        LITQToSITK,
        "litq",
        case_folder,
        tags=["StudyDate", "SliceThickness"],
        debug=debug,
    )

    summ = pd.read_csv("/s/datasets_bkp/litq/cases_summary.csv")

    parent = Path("/s/datasets_bkp/litq/dicom/")
    cases = list(parent.glob("*"))
# %%
    summ = pd.read_csv(D.cases_summary_fn)
    print(
        "Summary file {} exists. Set append to False if you want a fresh start".format(
            se
        )
    )

    existing = map(Path, list(summ.case_folder))
    new_cases = set(D.cases).difference(set(existing))
    taken_ids = list(summ.sitk_id)
    bb = list(map(int_from_string, taken_ids))
    largest_ind = max(bb)
    ref = list(range(largest_ind))
    gaps = list(set(ref).difference(bb))
    num_new_inds = len(new_cases) - len(gaps)
    starting_ind = largest_ind + 1
    new_inds = gaps + list(range(starting_ind, starting_ind + num_new_inds))
    cases_summary = [
        [case, D.generate_sitk_id(ind)] for case, ind in zip(new_cases, new_inds)
    ]

# %%
    logging.warning("This will get logged to a file")
# %%
    fn1 = "/media/ub/datasets_bkp2/normal/31no/DICOM/000047FB/AA8BFCEB/AA3D8AC2/000068E5/EE0A6B0B"
    fn2 = "/media/ub/datasets_bkp2/normal/31/DICOM/000040F5/AA399166/AA69ACA1/0000D667/EE0A437B"
    #     f = folders[0]
    h1 = dcmread(fn1)
    h2 = dcmread(fn2)

#
# %%
#         min_files=50
#         folder =Path("/s/datasets_bkp/litq/dicom/10/DICOM/")
#         folder = Path("/s/datasets_bkp/litqsmall/dicom/29")
#
# %%
#         fldrs = list(folder.rglob("*"))
#         ser_f =[fldr.parent for fldr in fldrs if fldr.is_file() ]
#         ser_f = set(ser_f)
#         ser_f = ser_f.difference([folder])
# %%
# #         folders =[]
# #         excluded_folders=[]
# #         aa =
# #         for x in ser.rglob("*"):
# #             if (x.is_file() and x.parent not in il.chain(folders,excluded_folders)):
# #                     candidate_fldr = x.parent
#                     nfiles = len(list(candidate_fldr.glob("*")))
#                                 folders.append(candidate_fldr)
#                     else:
#                         excluded_folders.append(candidate_fldr)
#
# %%
