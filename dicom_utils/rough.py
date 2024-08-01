# %%
from dicom_utils.dcm_to_sitk import *
import matplotlib.pyplot as plt


# %%
if __name__ == "__main__":
# %%
    parent =Path("/s/xnat_shadow/crc/") 
    fldr = Path("/s/xnat_shadow/crc/pending/images")
    fldr2 = Path("/s/xnat_shadow/crc/staging/images")
    files = list(fldr.glob("*"))
    files2 = list(fldr2.glob("*"))
    files = files+files2
    files.sort()
    names = [fn.name for fn in files]
    data = [[f,n] for f,n in zip(files,names)]
    df = pd.DataFrame(data,columns=['image_fn', 'name'])
    fn = parent/("notes2.csv")
    df.to_csv(fn,index=False)
# %%
    fn = "/s/insync/crc2/tmp/DICOM/00006849/AA003F6E/AA6D93E1/0000C4CE/FF229EC0.dcm"
    fn2 = "/s/insync/crc2/tmp/DICOM/00006849/AA003F6E/AA6D93E1/0000C4CE/FFF6864E.dcm"
    fn3 = "/s/insync/crc2/t2/DICOM/00009760/AA2A8CB6/AA17DA7A/0000E836/FFCBA0F7.dcm"

    dd1 = dcmread(fn)
    dd2 = dcmread(fn2)
    dd3 = dcmread(fn3)

    im = dd3.overlay_array(0x6000)
    plt.imshow(im)
    plt.show()
    aa = "/s/insync/crc2/tmp/DICOMDIR.dcm"
    ab = dcmread(aa)
# %%
