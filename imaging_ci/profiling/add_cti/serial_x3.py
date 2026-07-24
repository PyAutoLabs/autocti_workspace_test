"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - serial CTI is added to the image using a 2 `Trap` species model.
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
total_rows = 2000
shape_native = (total_rows, 2000)

dataset_type = "imaging_ci"

# dataset_label = "uniform"
dataset_label = "non_uniform"

dataset_name = "serial_x3"
# dataset_name = "serial_x3_seeded"

dataset_size = f"rows_{total_rows}"

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

The CTI model used by arCTIc to add CTI to the input image in the serial direction, which contains: 

 - 2 `Trap` species in the serial direction.
 - A simple CCD volume beta parametrization.
"""
serial_trap_0 = ac.TrapInstantCapture(density=0.07275, release_timescale=0.8)
serial_trap_1 = ac.TrapInstantCapture(density=0.21825, release_timescale=4.0)
serial_trap_2 = ac.TrapInstantCapture(density=6.54804, release_timescale=20.0)

serial_trap_list = [serial_trap_0, serial_trap_1, serial_trap_2]

serial_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)


"""
__Profile Settings__

Settings which dictate the speed of arctic of clocking and therefore will change the profiling results.
"""
serial_express = 2

"""
__Profile Normal__

The time to add CTI without the fast speed up.
"""
clocker = ac.Clocker2D(serial_express=serial_express)

cti = ac.CTI2D(serial_trap_list=serial_trap_list, serial_ccd=serial_ccd)

start = time.time()

image_slow = clocker.add_cti(data=image.native, cti=cti)

print(f"Clocking Time = {(time.time() - start)}")

"""
__Profile Fast__

The time to add CTI with the fast speed up.
"""
clocker = ac.Clocker2D(serial_express=serial_express, serial_fast_mode=True)


start = time.time()

image_fast = clocker.add_cti(data=image.native, cti=cti)

print(f"Clocking Time = {(time.time() - start)}")

print(np.max(np.abs(image_slow - image_fast)))

"""
Finished.
"""
