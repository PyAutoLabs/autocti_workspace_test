from astropy.io import fits
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
    "1-1_F",
    "1-1_G",
    "1-1_H",
    "6-4_E",
    "6-4_F",
    "6-4_G",
    "6-4_H",
]

parallel_overscan_list = []

for extension in extension_list:

    print(f"\n VALIDATION CHECKS FOR EXTENSION {extension}\n")

    dataset_name = f"ROS1_{extension}.fits"

    dataset_path = path.join("tvac", "dataset", ros_sequence, dataset_name)

    """
    Open the .fits data and extract the science / data headers, including the CCD ID and Quadrant ID.
    """
    data_hdulist = fits.open(dataset_path)

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
    which in this example uses the parallel overscan.
    
    The pixel indexes of this regions are updated to account for the rotation of the data performed above.
    """
    layout_2d = ac.Layout2DCI.from_euclid_fits_header(
        ext_header=data_header,
    )

    """
    Stash the parallel overscan to ensure all dataseta are identical at the end.
    """
    parallel_overscan_list.append(layout_2d.parallel_overscan)

    """
    Extract an array only containing the parallel overscan.
    
    This array should not contain any signal, and therefore have values near the bias level of ~2600e-.
    """
    array_2d_list = layout_2d.extract.parallel_overscan.array_2d_list_from(
        array=image_ci, pixels=(0, 20)
    )

    print(f"Parallel Overscan Min value (Expect ~2600e-) = {np.min(array_2d_list[0])}")

    """
    Extract an array containing the parallel overscan and the first column of the parallel FPR.
    
    This array should contain signal, and therefore have maximum values near ~3600e-
    """
    array_2d_list = layout_2d.extract.parallel_overscan.array_2d_list_from(
        array=image_ci, pixels=(-100, 20)
    )

    print(f"Parallel Overscan w/ FPR Max (Expect >3000e-) = {np.max(array_2d_list[0])}")
    print(f"CI Overscan w/ FPR Min (Expect ~2600e-) = {np.min(array_2d_list[0])}")


print(
    "\n Check that Parallel Overscan For Every Dataset Are Same for Every CCD / Quad ID:"
)

for parallel_overscan in parallel_overscan_list:

    print(f"Parallel Overscan: {parallel_overscan}")
