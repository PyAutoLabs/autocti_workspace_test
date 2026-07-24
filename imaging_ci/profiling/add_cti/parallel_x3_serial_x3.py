"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallle direction using the `CCD` class.
"""
import numpy as np
from os import path
import time

import autocti as ac

"""
__Dataset Paths__

The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/noise_map.fits'.
 - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/pre_cti_data.fits'.
"""
total_columns = 2000
shape_native = (2000, total_columns)

dataset_type = "imaging_ci"

dataset_label = "uniform"
# dataset_label = "non_uniform"


dataset_name = "parallel_x3"

dataset_size = f"columns_{total_columns}"

dataset_norm = f"norm_100"

filename = "pre_cti_data.fits"
# filename = "image.fits"

dataset_path = path.join(
    "dataset", dataset_type, dataset_label, dataset_name, dataset_size, dataset_norm
)

"""
__Layout__

The 2D shape of the image.
"""
image = ac.Array2D.from_fits(
    file_path=path.join(dataset_path, filename), hdu=0, pixel_scales=0.1
)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 2 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
"""
parallel_trap_0 = ac.TrapInstantCapture(
    density=0.34404498834887554, release_timescale=1.4499316054484241
)
parallel_trap_1 = ac.TrapInstantCapture(
    density=0.10817484460140586, release_timescale=6.2620610019940095
)

parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.7020370102952183, well_notch_depth=0.0, full_well_depth=200000.0
)

factor = 1.0

serial_trap_0 = ac.TrapInstantCapture(
    density=factor * 0.1019478899846178, release_timescale=1.2049949540017573
)
serial_trap_1 = ac.TrapInstantCapture(
    density=factor * 0.1757817374972805, release_timescale=5.2634561852450545
)
serial_trap_2 = ac.TrapInstantCapture(
    density=factor * 13.325436739291849, release_timescale=16.295735476619164
)

serial_trap_list = [serial_trap_0, serial_trap_1, serial_trap_2]

serial_ccd = ac.CCDPhase(
    well_fill_power=0.38350948311740113, well_notch_depth=0.0, full_well_depth=200000.0
)


"""
__Profile Settings__

Settings which dictate the speed of arctic of clocking and therefore will change the profiling results.
"""
parallel_express = 2
parallel_roe = ac.ROEChargeInjection()
# parallel_roe = ac.ROE()
serial_express = 2

"""
__Profile Normal__

The time to add CTI without the fast speed up.
"""
clocker = ac.Clocker2D(
    parallel_express=parallel_express,
    parallel_roe=parallel_roe,
    #  parallel_prune_frequency=0,
    parallel_fast_mode=True,
    #  serial_express=serial_express,
)

cti = ac.CTI2D(
    parallel_trap_list=parallel_trap_list,
    parallel_ccd=parallel_ccd,
)

image_with_parallel = clocker.add_cti(data=image.native, cti=cti)

clocker = ac.Clocker2D(
    serial_express=serial_express,
)

cti = ac.CTI2D(serial_trap_list=serial_trap_list, serial_ccd=serial_ccd)

# image_with_parallel[image_with_parallel < 1.0e-5] = 0.0

row_index = 1826

image_extract = image_with_parallel[row_index, :]

image_pass = np.zeros(shape=(1, 2000))
image_pass[0, :] = np.asarray(image_extract)

image_pass = ac.Array2D.no_mask(values=image_pass, pixel_scales=0.1)

file_path = path.join(dataset_path, "slow_serial.fits")

image_pass.output_to_fits(file_path=file_path)

for row_index in range(image_with_parallel.shape[1]):

    image_extract = image_with_parallel[row_index, :]

    image_pass = np.zeros(shape=(1, 2000))
    image_pass[0, :] = np.asarray(image_extract)

    image_pass = ac.Array2D.no_mask(values=image_pass, pixel_scales=0.1)

    start = time.time()

    image_with_parallel_serial = clocker.add_cti(data=image_pass, cti=cti)

    print(
        f"Clocking Time = {row_index} : {np.max(image_pass)} : {(time.time() - start)}"
    )

# file_path = path.join(dataset_path, "with_parallel_serial_cti.fits")
#
# image_slow.output_to_fits(file_path=file_path)

"""
__Profile Fast__

The time to add CTI with the fast speed up.
"""
# clocker = ac.Clocker2D(
#     parallel_express=parallel_express,
#     parallel_roe=parallel_roe,
#     parallel_fast_mode=True,
#     parallel_prune_frequency=0,
# )
#
#
# start = time.time()
#
# image_fast = clocker.add_cti(data=image.native, cti=cti)
#
# print(f"Clocking Time = {(time.time() - start)}")
#
#
# print(np.max(np.abs(image_slow - image_fast)))

"""
Finished.
"""
