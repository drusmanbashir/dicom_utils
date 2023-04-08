from fran.inference.scoring import Path
import itk
import SimpleITK as sitk
from dcm_to_sitk import *
import re
from fran.utils.sitk_utils import align_sitk_imgs, create_sitk_as
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


def multiprocess_convert_drli(dataset_id, case_folder,output_img_folder,output_masks_folder,ext,target_label,overwrite=False):

    case_id = dataset_id+"_"+folder_to_case_id(case_folder)
    img_folder = case_folder/("im_1")
    partial_mask_fn =  case_folder/("im_2/x0000.dcm")
    outnames = [folder/(case_id+ext) for folder in [output_img_folder,output_masks_folder]]
    assert partial_mask_fn.exists(),"File name {0} does not exist. Check {1}".format(partial_mask_fn,case_folder)
    if overwrite==False:
        assert not any([fn.exists() for fn in outnames]),"Files {}\nalready exist. Skipping.. ".format(outnames)
    img = dcmfolder_to_sitk(img_folder)
    mask = dcm_segmentation(partial_mask_fn)

    imsize = img.GetSize()[::-1]
    full_mask = np.zeros(imsize,dtype=np.uint8)
    mask_only = sitk.GetArrayFromImage(mask)
    if target_label:
        mask_only[mask_only>0]=1
    z_start = get_diff_z(img,mask)
    z_end = int(z_start+mask_only.shape[0])
    full_mask[z_start:z_end,:,:] = mask_only

    final_mask = create_sitk_as(img,full_mask)

    for img_, outname in zip([img,final_mask], outnames):
        sitk.WriteImage(img_,outname)

class ConvertDRLIToSITK():
    '''
    Masks are as specific segmentation dicom File   
    Each mask has lesions stored as sepoaraate islands requiring collation.
    '''
    def __init__(self,dicom_folder:Path, output_folder, ext='nrrd'):
        self.ext = "."+ext
        self.dataset_id = 'drli'
        self.case_folders = dicom_folder.glob("*")
        self.output_img_folder = output_folder/("images")
        self.output_masks_folder = output_folder/("masks")
        maybe_makedirs([self.output_img_folder,self.output_masks_folder])
    def process(self,target_label=None,overwrite=False,debug=False):
        args = [[self.dataset_id, case_folder,self.output_img_folder,self.output_masks_folder,self.ext,target_label,overwrite] for case_folder in self.case_folders]
        res = multiprocess_multiarg(func=multiprocess_convert_drli,arguments=args,num_processes=8,debug=debug)



# %%

if __name__ == "__main__":

    output_folder =Path("/media/ub2/datasets/drli/sitk")
    parent = Path("/media/ub2/datasets/drli/dicom")
    C = ConvertDRLIToSITK(parent,output_folder)
    C.process(target_label = 2, overwrite=True)
# %%
    dataset_id = 'drli'
    ext = '.nrrd'
    overwrite=True
    output_img_folder = output_folder/("images")
    output_masks_folder = output_folder/("masks")
    case_folder = Path("/media/ub2/datasets/drli/dicomtmp/AI-DRLI-048")
    case_id = dataset_id+"_"+folder_to_case_id(case_folder)
    target_label=2
# %%
    img_folder = case_folder/("im_1")
    partial_mask_fn =  case_folder/("im_2/x0000.dcm")
    outnames = [folder/(case_id+ext) for folder in [output_img_folder,output_masks_folder]]
    assert partial_mask_fn.exists(),"File name {0} does not exist. Check {1}".format(partial_mask_fn,case_folder)
    if overwrite==False:
        assert not any([fn.exists() for fn in outnames]),"Files {}\nalready exist. Skipping.. ".format(outnames)
    img = dcmfolder_to_sitk(img_folder)
    mask = dcm_segmentation(partial_mask_fn)

    imsize = img.GetSize()[::-1]
    full_mask = np.zeros(imsize,dtype=np.uint8)
    mask_only = sitk.GetArrayFromImage(mask)
    if target_label:
        mask_only[mask_only>0]=1
    z_start = get_diff_z(img,mask)
    z_end = int(z_start+mask_only.shape[0])
    full_mask[z_start:z_end,:,:] = mask_only
    final_mask = create_sitk_as(img,full_mask)

    final_mask  = sitk.GetImageFromArray(full_mask)
    final_mask.SetDirection(img.GetDirection())
    dirfilter = itk.OrientIm
    final_mask = set_metadata_as(final_mask,img)
# %%
    for img_, outname in zip([img,final_mask], outnames):
        sitk.WriteImage(img_,outname)

# %%

