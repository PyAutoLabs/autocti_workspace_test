from astropy.io import fits
from os import path


# dataset_path = path.join("tvac", "dataset", "ros1")
# mef_in = path.join(dataset_path, "20221012_171202_gnd0_VIS_SPW1N_31286.bin_01_01.fits" )

# dataset_path = path.join("tvac", "dataset", "ros5")
# mef_in = path.join(dataset_path, "20221012_175601_gnd0_VIS_SPW1N_33481.bin_01_01.fits" )

# dataset_path = path.join("tvac", "dataset", "ros10")
# mef_in = path.join(dataset_path, "20221012_185601_gnd0_VIS_SPW1N_39182.bin_01_01.fits" )

dataset_path = path.join("tvac", "dataset", "ros15")
mef_in = path.join(dataset_path, "20221012_194501_gnd0_VIS_SPW1N_37355.bin_01_01.fits")

mef_hdul = fits.open(mef_in, memmap=False)
prim_hdr = mef_hdul[0].header

ext_list = [1, 2, 3, 4, 141, 142, 143, 144]

for ext in ext_list:

    hdu_in = mef_hdul[int(ext)]
    quad_dat = hdu_in.data.astype("float32")
    quad_hdr = hdu_in.header

    ccd_id = quad_hdr["CCDID"]
    quadrant_id = quad_hdr["QUADID"]

    sci_name = f"ROS_{ccd_id}_{quadrant_id}.fits"

    primary_hdu = fits.PrimaryHDU(data=None, header=prim_hdr)
    image_hdu = fits.ImageHDU(data=quad_dat, header=quad_hdr)
    hdul_out = fits.HDUList([primary_hdu, image_hdu])
    hdul_out.writeto(path.join(dataset_path, sci_name), overwrite=True)
