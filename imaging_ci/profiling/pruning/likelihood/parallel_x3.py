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
dataset_label = "uniform"
dataset_name = "parallel_x3"
dataset_size = f"columns_{total_columns}"

dataset_path = path.join(
    "dataset", dataset_type, dataset_label, dataset_name, dataset_size
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
norm_list = [200000.0]
# 250.0, 500.0, 1000.0, 5000.0, 10000.0, 30000.0, 200000.0]

layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=regions_list,
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

imaging_ci_list = []

for norm in norm_list:

    imaging_ci_list.append(
        ac.ImagingCI.from_fits(
            image_path=path.join(dataset_path, f"norm_{int(norm)}", f"image.fits"),
            noise_map_path=path.join(
                dataset_path, f"norm_{int(norm)}", f"noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_path, f"norm_{int(norm)}", f"pre_cti_data.fits"
            ),
            layout=layout,
            pixel_scales=0.1,
        )
    )

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 3 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
"""

parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)
parallel_trap_2 = af.Model(ac.TrapInstantCapture)

parallel_trap_0.density = af.GaussianPrior(mean=0.07275, sigma=1.0)
parallel_trap_0.release_timescale = af.GaussianPrior(mean=0.8, sigma=1.0)

parallel_trap_1.density = af.GaussianPrior(mean=0.21825, sigma=1.0)
parallel_trap_1.release_timescale = af.GaussianPrior(mean=4.0, sigma=1.0)

parallel_trap_2.density = af.GaussianPrior(mean=6.54804, sigma=1.0)
parallel_trap_2.release_timescale = af.GaussianPrior(mean=20.0, sigma=1.0)

parallel_trap_list = [parallel_trap_0, parallel_trap_1, parallel_trap_2]

parallel_ccd = af.Model(ac.CCDPhase)

parallel_ccd.well_fill_power = 0.58
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(
        ac.CTI2D, parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd
    )
)

instance = model.instance_from_prior_medians()

"""
__Profile Settings__

Settings which dictate the speed of arctic of clocking and therefore will change the profiling results.
"""
parallel_express = 2
parallel_roe = ac.ROEChargeInjection()

"""
__Pruning Settings__
"""
for imaging in imaging_ci_list:

    clocker = ac.Clocker2D(
        parallel_express=parallel_express,
        parallel_roe=parallel_roe,
        parallel_prune_frequency=0,
    )

    analysis = ac.AnalysisImagingCI(dataset=imaging, clocker=clocker)

    """
    __Profile Normal__
    
    The time to add CTI without the fast speed up.
    """
    no_pruning_lh = analysis.log_likelihood_function(instance=instance)

    """
    __Pruning Settings__
    """
    clocker = ac.Clocker2D(
        parallel_express=parallel_express,
        parallel_roe=parallel_roe,
        parallel_prune_n_electrons=1e-8,
        parallel_prune_frequency=20,
    )

    analysis = ac.AnalysisImagingCI(dataset=imaging, clocker=clocker)

    """
    __Profile Normal__
    
    The time to add CTI without the fast speed up.
    """

    pruning_lh = analysis.log_likelihood_function(instance=instance)
    print(f"No (Pruning / No Pruning)= {pruning_lh} | {no_pruning_lh}")
