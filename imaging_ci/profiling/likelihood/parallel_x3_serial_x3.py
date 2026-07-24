"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - parallel CTI is added to the image using a 2 `Trap` species model.
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
total_columns = 2000
shape_native = (2000, total_columns)

dataset_type = "imaging_ci"

dataset_label = "non_uniform"
dataset_name = "parallel_x3"

dataset_size = f"columns_{total_columns}"

dataset_norm = f"norm_10000"
# dataset_norm = f"norm_30000"
# dataset_norm = f"norm_200000"

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
regions_list = [
    (30, 60, serial_prescan[3], serial_overscan[2]),
    (360, 390, serial_prescan[3], serial_overscan[2]),
    (690, 720, serial_prescan[3], serial_overscan[2]),
    (1020, 1050, serial_prescan[3], serial_overscan[2]),
    (1350, 1380, serial_prescan[3], serial_overscan[2]),
    (1680, 1710, serial_prescan[3], serial_overscan[2]),
]

"""
__Layout__

The 2D shape of the image.
"""

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

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 3 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
"""

# parallel_trap_0 = af.Model(ac.TrapInstantCapture)
# parallel_trap_1 = af.Model(ac.TrapInstantCapture)
# parallel_trap_2 = af.Model(ac.TrapInstantCapture)
#
# parallel_trap_0.density = af.GaussianPrior(mean=0.07275, sigma=1.0)
# parallel_trap_1.density = af.GaussianPrior(mean=0.21825, sigma=1.0)
# parallel_trap_2.density = af.GaussianPrior(mean=6.54804, sigma=1.0)
#
# parallel_trap_0.release_timescale = af.GaussianPrior(mean=0.8, sigma=1.0)
# parallel_trap_1.release_timescale = af.GaussianPrior(mean=4.0, sigma=1.0)
# parallel_trap_2.release_timescale = af.GaussianPrior(mean=20.0, sigma=1.0)
#
# parallel_traps = [parallel_trap_0, parallel_trap_1, parallel_trap_2]
#
# parallel_ccd = af.Model(ac.CCDPhase)
#
# parallel_ccd.well_fill_power = 0.58
# parallel_ccd.well_notch_depth = 0.0
# parallel_ccd.full_well_depth = 200000.0
#
# serial_trap_0 = af.Model(ac.TrapInstantCapture)
# serial_trap_1 = af.Model(ac.TrapInstantCapture)
# serial_trap_2 = af.Model(ac.TrapInstantCapture)
#
# serial_trap_0.density = af.GaussianPrior(mean=0.07275, sigma=1.0)
# serial_trap_1.density = af.GaussianPrior(mean=0.21825, sigma=1.0)
# serial_trap_2.density = af.GaussianPrior(mean=6.54804, sigma=1.0)
#
# serial_trap_0.release_timescale = af.GaussianPrior(mean=0.8, sigma=1.0)
# serial_trap_1.release_timescale = af.GaussianPrior(mean=4.0, sigma=1.0)
# serial_trap_2.release_timescale = af.GaussianPrior(mean=20.0, sigma=1.0)
#
# serial_trap_list = [serial_trap_0, serial_trap_1, serial_trap_2]
#
# serial_ccd = af.Model(ac.CCDPhase)
#
# serial_ccd.well_fill_power = 0.58
# serial_ccd.well_notch_depth = 0.0
# serial_ccd.full_well_depth = 200000.0


parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)

parallel_trap_0.density = af.GaussianPrior(mean=0.34404498834887554, sigma=1.0)
parallel_trap_1.density = af.GaussianPrior(mean=0.10817484460140586, sigma=1.0)

parallel_trap_0.release_timescale = af.GaussianPrior(mean=1.4499316054484241, sigma=1.0)
parallel_trap_1.release_timescale = af.GaussianPrior(mean=6.2620610019940095, sigma=1.0)

parallel_traps = [parallel_trap_0, parallel_trap_1]

parallel_ccd = af.Model(ac.CCDPhase)

parallel_ccd.well_fill_power = 0.7020370102952183
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

serial_trap_0 = af.Model(ac.TrapInstantCapture)
serial_trap_1 = af.Model(ac.TrapInstantCapture)
serial_trap_2 = af.Model(ac.TrapInstantCapture)

serial_trap_0.density = af.GaussianPrior(mean=0.1019478899846178, sigma=1.0)
serial_trap_1.density = af.GaussianPrior(mean=0.1757817374972805, sigma=1.0)
serial_trap_2.density = af.GaussianPrior(mean=13.325436739291849, sigma=1.0)

serial_trap_0.release_timescale = af.GaussianPrior(mean=1.2049949540017573, sigma=1.0)
serial_trap_1.release_timescale = af.GaussianPrior(mean=5.2634561852450545, sigma=1.0)
serial_trap_2.release_timescale = af.GaussianPrior(mean=16.295735476619164, sigma=1.0)

serial_trap_list = [serial_trap_0, serial_trap_1, serial_trap_2]

serial_ccd = af.Model(ac.CCDPhase)

serial_ccd.well_fill_power = 0.38350948311740113
serial_ccd.well_notch_depth = 0.0
serial_ccd.full_well_depth = 200000.0


model = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=parallel_traps,
        parallel_ccd=parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )
)

instance = model.instance_from_prior_medians()

"""
__Profile Settings__

Settings which dictate the speed of arctic of clocking and therefore will change the profiling results.
"""
parallel_express = 2
parallel_roe = ac.ROEChargeInjection()
serial_express = 2

clocker = ac.Clocker2D(
    parallel_express=parallel_express,
    parallel_roe=parallel_roe,
    serial_express=serial_express,
    parallel_fast_mode=True,
    #   serial_prune_n_electrons=1e-7,
    #   serial_prune_frequency=10
)

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

print(f"LH Evaluation Time = {time_normal}")

# """
# __Profile Fast__
#
# The time to add CTI with the fast speed up.
# """
# clocker = ac.Clocker2D(
#     parallel_express=parallel_express,
#     parallel_roe=parallel_roe,
#     serial_express=serial_express,
#     parallel_fast_mode=True,
# )
#
# analysis = ac.AnalysisImagingCI(dataset=imaging, clocker=clocker)
#
# start = time.time()
#
# for i in range(repeats):
#
#     print(analysis.log_likelihood_function(instance=instance))
#
# time_fast = (time.time() - start) / repeats
#
# print(f"LH Evaluation Time Fast Mode = {time_fast}")
"""
Finished.
"""
