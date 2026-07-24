"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - serial CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallle direction using the `CCD` class.
"""
from os import path
import time

import autofit as af
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
dataset_label = "uniform"
dataset_name = "serial_x3"
dataset_size = f"rows_{total_rows}"

dataset_norm = f"norm_100"

dataset_path = path.join(
    "dataset", dataset_type, dataset_label, dataset_name, dataset_size, dataset_norm
)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D(
    (shape_native[0] - 1, shape_native[0], 1, shape_native[1] - 1)
)
serial_prescan = ac.Region2D((0, shape_native[0], 0, 1))
serial_overscan = ac.Region2D(
    (0, shape_native[0] - 1, shape_native[1] - 1, shape_native[1])
)

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
regions_list = [(0, shape_native[0] - 1, serial_prescan[3], serial_overscan[2])]

"""
__Layout__

The 2D shape of the image.
"""
norm = 100

layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=regions_list,
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

imaging = ac.ImagingCI.from_fits(
    image_path=path.join(dataset_path, f"image.fits"),
    noise_map_path=path.join(dataset_path, f"noise_map.fits"),
    pre_cti_data_path=path.join(dataset_path, f"pre_cti_data.fits"),
    layout=layout,
    pixel_scales=0.1,
)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the serial direction, which contains: 

 - 3 `Trap` species in the serial direction.
 - A simple CCD volume beta parametrization.
"""

serial_trap_0 = af.Model(ac.TrapInstantCapture)
serial_trap_1 = af.Model(ac.TrapInstantCapture)
serial_trap_2 = af.Model(ac.TrapInstantCapture)

serial_trap_0.density = af.GaussianPrior(mean=0.07275, sigma=1.0)
serial_trap_1.density = af.GaussianPrior(mean=0.21825, sigma=1.0)
serial_trap_2.density = af.GaussianPrior(mean=6.54804, sigma=1.0)

serial_trap_0.release_timescale = af.GaussianPrior(mean=0.8, sigma=1.0)
serial_trap_1.release_timescale = af.GaussianPrior(mean=4.0, sigma=1.0)
serial_trap_2.release_timescale = af.GaussianPrior(mean=20.0, sigma=1.0)

serial_trap_list = [serial_trap_0, serial_trap_1, serial_trap_2]

serial_ccd = af.Model(ac.CCDPhase)

serial_ccd.well_fill_power = 0.58
serial_ccd.well_notch_depth = 0.0
serial_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(ac.CTI2D, serial_trap_list=serial_trap_list, serial_ccd=serial_ccd)
)

instance = model.instance_from_prior_medians()

"""
__Profile Settings__

Settings which dictate the speed of arctic of clocking and therefore will change the profiling results.
"""
serial_express = 2

clocker = ac.Clocker2D(serial_express=serial_express)

analysis = ac.AnalysisImagingCI(dataset=imaging, clocker=clocker)

repeats = 1

"""
__Profile Normal__

The time to add CTI without the fast speed up.
"""
start = time.time()
for i in range(repeats):

    print(analysis.log_likelihood_function(instance=instance))

time_normal = (time.time() - start) / repeats

print(f"LH Evaluation Time Serial = {time_normal}")

"""
__Profile Fast__

The time to add CTI with the fast speed up.
"""
clocker = ac.Clocker2D(serial_express=serial_express, serial_fast_mode=True)

analysis = ac.AnalysisImagingCI(dataset=imaging, clocker=clocker)

start = time.time()

for i in range(repeats):

    print(analysis.log_likelihood_function(instance=instance))

time_fast = (time.time() - start) / repeats

print(f"LH Evaluation Time Serial = {time_fast}")
"""
Finished.
"""
