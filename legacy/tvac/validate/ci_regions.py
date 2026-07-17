from astropy.io import fits
import copy
import numpy as np
from os import path

import autocti as ac
import autocti.plot as aplt

"""
Setup the dataset path, including a specification of the charge Injection ROS sequence and the CCD id / Quadrant ID 
extension.
"""
ros_sequence = "ros1"
# ros_sequence = "ros5"
# ros_sequence = "ros10"
# ros_sequence = "ros15"

extension_list = [
    "1-1_E",
    #  "1-1_F",
    #   "1-1_G",
    #  "1-1_H",
    #   "6-4_E",
    #   "6-4_F",
    #  "6-4_G",
    #   "6-4_H",
]

# extension_list = [extension_list[4]]

region_list = []

for extension in extension_list:

    print(f"\n VALIDATION CHECKS FOR EXTENSION {extension}\n")

    dataset_name = f"ROS_{extension}.fits"

    dataset_path = path.join("tvac", "dataset", ros_sequence)

    """
    Open the .fits data and extract the science / data headers, including the CCD ID and Quadrant ID.
    """
    data_hdulist = fits.open(path.join(dataset_path, dataset_name))

    sci_header = data_hdulist[0].header
    data_header = data_hdulist[1].header

    data = data_hdulist[1].data

    ccd_id = data_header["CCDID"]
    quadrant_id = data_header["QUADID"]

    print(f"CCD ID = {ccd_id}")
    print(f"Quadrant ID = {quadrant_id}")

    """
    The science header does not contain the CCDID an QUADID entires which are required, thus we copy them over.
    """
    data_header["CI_IJON"] = sci_header["CI_IJON"]
    data_header["CI_IJOFF"] = sci_header["CI_IJOFF"]
    data_header["CI_VSTAR"] = sci_header["CI_VSTAR"]
    data_header["CI_VEND"] = sci_header["CI_VEND"]

    """
    Set up the 2D dataset, including rotations based on the CCD ID and Quadrant ID which mean that CTI clocking is
    the same direction for all datasets.
    """
    image_ci = ac.euclid.Array2DEuclid.from_fits_header(
        array=data.astype("float"), ext_header=data_header
    )

    """
    Set up the layout of the CCD, which contains integer pixel indexes of where certain regions on the CCD are located, 
    which in this example uses the ci regions.
    
    The pixel indexes of this regions are updated to account for the rotation of the data performed above.
    """
    layout_2d = ac.Layout2DCI.from_euclid_fits_header(
        ext_header=data_header,
    )

    """
    Stash the ci regions to ensure all dataseta are identical at the end.
    """
    region_list.append(layout_2d.region_list)

    """
    Extract the size of the charge injection regions across the rows (e.g. the size of the parallel FPR) and the 
    columns (e.g. the size of the serial FPR).
    """
    ci_size_across_rows = layout_2d.region_list[0][1] - layout_2d.region_list[0][0]
    ci_size_across_columns = layout_2d.region_list[0][3] - layout_2d.region_list[0][2]

    """
    Use the parallel FPR extraction method to extract an array containing only the parallel FPR. 
    
    This array should contain signal, and therefore have values above 3600-e
    """
    array_2d_list = layout_2d.extract.parallel_fpr.array_2d_list_from(
        array=image_ci, pixels=(0, ci_size_across_rows)
    )

    for i, arr_2d in enumerate(array_2d_list):

        print(
            f"Parallel FPR Min value (Expect >3000e-) [region {i}] = {np.min(arr_2d)}"
        )
        print(
            f"Parallel FPR Max value (Expect >3000e-) [region {i}] = {np.max(arr_2d)}"
        )

    """
    Repeat the extraction above, but with one extra pixel that is in front of the parallel FPR, meaning the minimum
    value should drop to ~2600e-.
    """
    array_2d_list = layout_2d.extract.parallel_fpr.array_2d_list_from(
        array=image_ci, pixels=(-1, ci_size_across_rows)
    )

    for i, arr_2d in enumerate(array_2d_list):
        print(
            f"Parallel FPR w/ 1 pixel in front Min value (Expect ~2600e-) [region {i}] = {np.min(arr_2d)}"
        )

    """
    Repeat the extraction above, but with one extra pixel that includes the parallel EPER, meaning the minimum
    value should drop to ~2600e-.
    """
    array_2d_list = layout_2d.extract.parallel_fpr.array_2d_list_from(
        array=image_ci, pixels=(0, ci_size_across_rows + 1)
    )

    for i, arr_2d in enumerate(array_2d_list):
        print(
            f"Parallel FPR w/ 1 pixel of EPER Min value (Expect ~2600e-) [region {i}] = {np.min(arr_2d)}"
        )

    """
    Extract based on the serial FPR, with one extra pixel that is in front of the serial FPR, meaning the minimum
    value should drop to ~2600e-.
    """
    array_2d_list = layout_2d.extract.serial_fpr.array_2d_list_from(
        array=image_ci, pixels=(-1, ci_size_across_columns)
    )

    for i, arr_2d in enumerate(array_2d_list):
        print(
            f"Serial FPR w/ 1 pixel in front Min value (Expect ~2600e-) [region {i}] = {np.min(arr_2d)}"
        )

    """
    Now repeat but including the serial EPER instead.
    """
    array_2d_list = layout_2d.extract.serial_fpr.array_2d_list_from(
        array=image_ci, pixels=(0, ci_size_across_columns + 1)
    )

    for i, arr_2d in enumerate(array_2d_list):
        print(
            f"Serial FPR w/ 1 pixel EPER Min value (Expect ~2600e-) [region {i}] = {np.min(arr_2d)}"
        )

    image_ci_copy = copy.copy(image_ci.native)

    for region in layout_2d.region_list:

        image_ci_copy[region.slice] = 0.0

    print(f"Data Max After Region Remove = {np.max(image_ci_copy)}")

    index = np.unravel_index(image_ci_copy.argmax(), image_ci_copy.shape)

    image_ci_copy = ac.Array2D.no_mask(values=image_ci_copy, pixel_scales=0.1)

    output = aplt.Output(
        path=dataset_path,
        filename=dataset_name,
        format="png",
    )

    mat_plot_2d = aplt.MatPlot2D(output=output)

    array_2d_plotter = aplt.Array2DPlotter(array=image_ci_copy, mat_plot_2d=mat_plot_2d)
    array_2d_plotter.figure_2d()

print("\n Check that CI Regions For Every Dataset Are Same for Every CCD / Quad ID:")

for region in region_list:

    print(f"CI Regions: {region}")
