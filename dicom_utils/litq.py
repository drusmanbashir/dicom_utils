

# %%
import re
import itertools as il

from fastai.data.core import test_close, test_eq
from dcm_to_sitk import *
import pandas as pd
def int_from_string(s):
        a=re.search('\d+',s)
        b = int(a.group())
        return b



# %%
if __name__ == "__main__":
    parent = Path("/s/datasets_bkp/litq/dicom/")

    D = LITQToSITK("litq", parent,tags= ['StudyDate'],rename_sitk=True,start_ind=10)
    D.save_cases_summary()

    debug = False
    results = mp_dataset(LITQToSITK,'litqsmall',parent,tags =None ,debug=debug,overwrite=False)
# %%
    cases = list(map(lambda x:str(x),parent.glob("*")))

