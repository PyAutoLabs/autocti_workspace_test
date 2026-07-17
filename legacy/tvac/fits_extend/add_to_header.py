from astropy.io import fits
import os

file_list = os.listdir(".")
file_list = list(
    filter(None, [file if "_gnd0_" in file else None for file in file_list])
)

for i, file in enumerate(file_list):

    data_hdulist = fits.open(file)

    sci_header = data_hdulist[0].header
    data_header = data_hdulist[1].header

    fits.setval(file, "CCDID", value=data_header["CCDID"])
    fits.setval(file, "QUADID", value=data_header["QUADID"])

    fits.setval(file, "OBS_ID", value=i)
