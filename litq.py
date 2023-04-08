
# %%
import re
import itertools as il

from fastai.data.core import test_close, test_eq
from dcm_to_sitk import *
import pandas as pd
# %%
class LITQToSITK(DCMToSITK):
    def __init__(self,dataset_name,parent, tags=['StudyDate'], sitk_ext='.nrrd',sitk_folder=None,rename_sitk=True,*args,**kwargs):
        super().__init__(dataset_name, parent, tags=tags, sitk_ext=sitk_ext,sitk_folder=sitk_folder,rename_sitk=rename_sitk,*args,**kwargs)
    def create_cases_summary(self):
        _cases = list(self.parent.glob("*"))
        colnames= ['case_folder','sitk_id'] if self.rename_sitk ==True else ['case_folder']
        if self.rename_sitk==True:
            self.cases_summary = []
            self.cases_summary= [[case, self.generate_sitk_id()] for case in _cases if case.is_dir()]
        else:
            self.cases_summary = _cases
        self.cases_summary = pd.DataFrame(self.cases_summary,columns=colnames)
        print(self.cases_summary)



# %%
if __name__ == "__main__":
    parent = Path("/s/datasets_bkp/litmp/DICOM/")
    D = LITQToSITK("litqsmall", parent,tags= None,rename_sitk=True,start_ind=0)
    D.save_cases_summary()

    debug = False
    results = mp_dataset(LITQToSITK,'litqsmall',parent,tags =None ,debug=debug)
# %%
