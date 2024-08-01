# %%
from pathlib import Path
import os

from dicom_utils.dcm_to_sitk import delete_unwanted_files_folders
from dicom_utils.metadata import dcm_id_fromfoldername
from fran.utils.helpers import multiprocess_multiarg, pp


if __name__ == "__main__":
    

    fldrs = Path("/s/datasets_bkp/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC")

    delete_unwanted_files_folders(fldrs)
# %%
    since_day=1 # only process folders added after this day of the mth
    import time
    cases = list(fldrs.glob("*"))
    sorted(cases,key=os.path.getmtime)
    files_from_17=[]
    for file_path in cases:
        timestamp_str = time.strftime(  '%m/%d/%Y :: %H:%M:%S',
                                    time.gmtime(os.path.getmtime(file_path))) 
        gg = time.gmtime(os.path.getmtime(file_path)) 
        if gg.tm_mday>since_day:
            files_from_17.append(file_path)
        print(timestamp_str, ' -->', file_path) 
    print("Files to change patient_id based on folder {0}".format(str(len(files_from_17))))
    pp(files_from_17)
# %%
    debug=False
    args = [[case] for case in files_from_17]
    multiprocess_multiarg(dcm_id_fromfoldername,args,debug=debug)

# %%

