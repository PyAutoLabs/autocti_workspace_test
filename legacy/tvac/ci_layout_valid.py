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

print(image_ci.shape_native)

print(layout_2d.region_list)

CI_IJSIZE = layout_2d.region_list[0][1] - layout_2d.region_list[0][0]

fpr_array_2d_list = layout_2d.extract.parallel_fpr.array_2d_list_from(
    array=image_ci, pixels=(0, CI_IJSIZE)
)

print(
    "Parallel FPR: Charge Injection Region Extract min / max (Expect 3000e- -. 4000e-)"
)

for i, arr_2d in enumerate(fpr_array_2d_list):

    print(f"CI Region {i} max = {np.max(arr_2d)}")
    print(f"CI Region {i} min = {np.min(arr_2d)}")

print()
print(
    "Parallel FPR: Charge Injection Region Extract Overshoot Above min / max (Expect ~2600e-)"
)

fpr_array_2d_list = layout_2d.extract.parallel_fpr.array_2d_list_from(
    array=image_ci, pixels=(0, CI_IJSIZE + 1)
)

for i, arr_2d in enumerate(fpr_array_2d_list):

    print(f"CI Region {i} min = {np.min(arr_2d)}")

print()
print(
    "Parallel FPR: Charge Injection Region Extract Overshoot Below min / max (Expect ~2600e-)"
)

fpr_array_2d_list = layout_2d.extract.parallel_fpr.array_2d_list_from(
    array=image_ci, pixels=(-1, CI_IJSIZE + 1)
)

for i, arr_2d in enumerate(fpr_array_2d_list):

    print(f"CI Region {i} min = {np.min(arr_2d)}")

print()

CI_IJSIZE = layout_2d.region_list[0][3] - layout_2d.region_list[0][2]

fpr_array_2d_list = layout_2d.extract.serial_fpr.array_2d_list_from(
    array=image_ci, pixels=(0, CI_IJSIZE)
)

print("Serial FPR: Charge Injection Region Extract min / max (Expect 3000e- -. 4000e-)")

for i, arr_2d in enumerate(fpr_array_2d_list):
    print(f"CI Region {i} max = {np.max(arr_2d)}")
    print(f"CI Region {i} min = {np.min(arr_2d)}")

print()
print(
    "Serial FPR: Charge Injection Region Extract Overshoot Above min / max (Expect ~2600e-)"
)

fpr_array_2d_list = layout_2d.extract.serial_fpr.array_2d_list_from(
    array=image_ci, pixels=(0, CI_IJSIZE + 1)
)

for i, arr_2d in enumerate(fpr_array_2d_list):
    print(f"CI Region {i} min = {np.min(arr_2d)}")

print()
print(
    "Serial FPR: Charge Injection Region Extract Overshoot Below min / max (Expect ~2600e-)"
)

fpr_array_2d_list = layout_2d.extract.serial_fpr.array_2d_list_from(
    array=image_ci, pixels=(-1, CI_IJSIZE + 1)
)

for i, arr_2d in enumerate(fpr_array_2d_list):
    print(f"CI Region {i} min = {np.min(arr_2d)}")

print()


# import copy
#
# image_ci_copy = copy.copy(image_ci.native)
#
# for region in layout_2d.region_list:
#
#     image_ci_copy[region.slice] = 0.0
#
# print(f"Data Max After Region Remove = {np.max(image_ci_copy)}")
#
# index = np.unravel_index(image_ci_copy.argmax(), image_ci_copy.shape)
#
# print(index)
#
# image_ci_copy = ac.Array2D.no_mask(values=image_ci_copy, pixel_scales=0.1)
#
# output = aplt.Output(
#     path=path.join("tvac", "dataset"),
#     filename="ROS1",
#     format="png",
# )
#
# mat_plot_2d = aplt.MatPlot2D(output=output)
#
# array_2d_plotter = aplt.Array2DPlotter(array=image_ci_copy, mat_plot_2d=mat_plot_2d)
# array_2d_plotter.figure_2d()
