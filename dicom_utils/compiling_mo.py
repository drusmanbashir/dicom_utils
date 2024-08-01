
# %%
from litq import *
from fran.utils.sitk_utils import get_metadata

# %%

parent = Path("/media/ub/datasets_bkp/litq/complete_cases/org")
all_nii = list(parent.rglob("*nii*"))

fn = all_nii[0]
mask_tags = ['ub','seg']
# %%
def extract_tags(fn):
    pat = r'(^[^-_]*)_\w*_(\d*).*'
    gps = re.match(pat,fn.name)
    if gps:
        return    *gps.groups(),fn
    else:
        return None, None, fn
# %%
out = [extract_tags(fn) for fn in all_nii]

# %%
case_ids,mask=[],[]
for nlist in out:
    fn = nlist[-1]
    aa = ['litq',]+list(nlist[:-1])
    case_id = "_".join(aa)
    case_ids.append(case_id)
    mask.append(any([m in fn.name.lower() for m in mask_tags]))
# %%
df= pd.DataFrame(out,columns = ['id','date','fn'])
df['case_id']= case_ids
df['mask'] = mask
# %%
case_ids = []
for n in range(len(df)):
    row = df.iloc[n]   
    case_id = '_'.join(['litq',*list(row[:2])])
    case_ids.append(case_id)
# %%
# %% [markdown]
## Creating output filenames
# %%
def create_output_path( row: pd.Series)->pd.Series:
    ext = row.fn.name.split('.')[1:]
    out_name = ".".join([row.case_id,*ext])
    category = "images","masks"
    output_path= parent.parent/(category[int(row['mask'])])/out_name
    return output_path
# %%
output_paths = [create_output_path(row) for i, row in df.iterrows()]
df['dest_paths'] = output_paths
df.to_csv(parent.parent/("case_ids.csv"),index=False)

# %%
def copy_src_to_dest(row:pd.Series)->None:
    src = row.fn
    dest = row.dest_paths
    if not dest.exists():
        shutil.copy(src,dest)
# %%
debug = False
args =[[y]  for x,y in df.iterrows()]
multiprocess_multiarg(copy_src_to_dest,args,debug=debug)
# %%
masks_folder = Path("/media/ub/datasets_bkp/litq/complete_cases/masks/")
masks = list(masks_folder.glob("*"))
masks_stray,masks_final=[],[]

imgs_final = [x.str_replace("masks","images") for x in masks if x.exists()]
masks_stray= [x.str_replace("masks","images") for x in masks if not x.exists()]
# %%
meta_imgs,meta_masks=[],[]
for mask_fn,img_fn in zip(masks,imgs_final):
    mask = sitk.ReadImage(mask_fn)
    img = sitk.ReadImage(img_fn)
    arrs = list(map(sitk.ReadImage,[img_fn,mask_fn]))
    meta_img,meta_mask = list(map(get_metadata,arrs))
    meta_imgs.append(meta_img)
    meta_masks.append(meta_mask)
# %%
matched = []
for meta_img,meta_mask in zip(meta_imgs,meta_masks):
    try:
        [test_close(a,b,eps=1e-3) for a,b in zip(meta_img,meta_mask)]
        matched.append("matched")
    except:
        matched.append("mismatch")
# %%
meta_str = ['size','spacing','direction']
meta_mask_cols,meta_img_cols = [["_".join([prefix,d]) for d in meta_str ] for prefix in ['masks','images']]
meta_mask_df , meta_img_df = [pd.DataFrame(x,columns=cols) for cols , x in zip([meta_mask_cols,meta_img_cols],[meta_masks,meta_imgs])]
df = pd.DataFrame(data = {'image_filenames':imgs_final,'mask_filenames':masks,'match':matched})
df_final = pd.concat([df,meta_mask_df,meta_img_df],axis = 1)
df_final.to_csv(parent.parent/("cases_metadata.csv"),index=False)
# %%
# %%
print(len(masks_stray))
# %%
preds_mo_folder = parent.parent/("predictions_mo")
predictions = preds_mo_folder.glob("*")
# %%
