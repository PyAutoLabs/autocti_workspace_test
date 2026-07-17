from astropy.io import fits
import numpy as np
from os import path

import autocti as ac
import autocti.plot as aplt

dataset_path = path.join("tvac", "dataset", "ros5")

extension = "1-1_E"

dataset_name = f"ROS1_{extension}.fits"

dataset_name = "EUC_VIS_EXP_000000-00-0.fits_sa_xt_bs_nl.001"

dataset_path = path.join(dataset_path, dataset_name)

data_hdulist = fits.open(dataset_path)

sci_header = data_hdulist[0].header
data_header = data_hdulist[1].header

data = data_hdulist[1].data

ccd_id = data_header["CCDID"]
quadrant_id = data_header["QUADID"]

print(f"CCD ID = {ccd_id}")
print(f"Quadrant ID = {quadrant_id}")

print(f"CI_IJON = {sci_header['CI_IJON']}")
print(f"CI_IJOFF = {sci_header['CI_IJOFF']}")
print(f"CI_VSTAR = {sci_header['CI_VSTAR']}")
print(f"CI_VEND = {sci_header['CI_VEND']}")

data_header["CI_IJON"] = sci_header["CI_IJON"]
data_header["CI_IJOFF"] = sci_header["CI_IJOFF"]
data_header["CI_VSTAR"] = sci_header["CI_VSTAR"]
data_header["CI_VEND"] = sci_header["CI_VEND"]

image_ci = ac.euclid.Array2DEuclid.from_fits_header(
    array=data.astype("float"), ext_header=data_header
)

layout_2d = ac.Layout2DCI.from_euclid_fits_header(
    ext_header=data_header,
)

image_1d_parallel_eper = layout_2d.extract.parallel_eper.binned_array_1d_from(
    array=image_ci, pixels=(0, 30)
)

mat_plot_1d = aplt.MatPlot1D(
    title=aplt.Title(label="Binned Parallel EPER trail"),
    ylabel=aplt.YLabel(label="signal (e-)"),
    output=aplt.Output(path="hi", filename="parallel_eper", format="png"),
)

array_1d_plotter = aplt.Array1DPlotter(
    y=image_1d_parallel_eper, mat_plot_1d=mat_plot_1d
)
array_1d_plotter.figure_1d()

image_1d_serial_eper = layout_2d.extract.serial_eper.binned_array_1d_from(
    array=image_ci, pixels=(0, 29)
)

mat_plot_1d = aplt.MatPlot1D(
    title=aplt.Title(label="Binned Serial EPER trail"),
    ylabel=aplt.YLabel(label="signal (e-)"),
    output=aplt.Output(path="hi", filename="serial_eper", format="png"),
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_serial_eper, mat_plot_1d=mat_plot_1d)
array_1d_plotter.figure_1d()
