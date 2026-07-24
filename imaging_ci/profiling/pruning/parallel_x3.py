"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallle direction using the `CCD` class.
"""
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
dataset_name = "parallel_x1_with_noise"
dataset_size = f"columns_{total_columns}"


norm_list = [100.0, 250.0, 500.0, 1000.0, 5000.0, 10000.0, 30000.0, 200000.0]

pre_cti_image_list = []

for norm in norm_list:

    dataset_path = path.join(
        "dataset",
        dataset_type,
        dataset_label,
        dataset_name,
        dataset_size,
        f"norm_{int(norm)}",
    )

    pre_cti_image_list.append(
        ac.Array2D.from_fits(
            file_path=path.join(dataset_path, "image.fits"), hdu=0, pixel_scales=0.1
        )
    )

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 2 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=0.07275, release_timescale=0.8)
parallel_trap_1 = ac.TrapInstantCapture(density=0.21825, release_timescale=4.0)
parallel_trap_2 = ac.TrapInstantCapture(density=6.54804, release_timescale=20.0)

parallel_trap_list = [parallel_trap_0, parallel_trap_1, parallel_trap_2]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)


"""
__Profile Settings__

Settings which dictate the speed of arctic of clocking and therefore will change the profiling results.
"""
parallel_express = 2
parallel_roe = ac.ROEChargeInjection()

"""
__Profile Normal__

The time to add CTI without the fast speed up.
"""
for pre_cti_image in pre_cti_image_list:

    clocker = ac.Clocker2D(
        parallel_express=parallel_express,
        parallel_roe=parallel_roe,
        parallel_prune_n_electrons=1.0e-8,
        parallel_prune_frequency=20,
    )

    start = time.time()

    cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

    clocker.add_cti(data=pre_cti_image.native, cti=cti)

    pruning_time = time.time() - start

    clocker = ac.Clocker2D(
        parallel_express=parallel_express,
        parallel_roe=parallel_roe,
        parallel_prune_n_electrons=-1e-12,
        parallel_prune_frequency=20,
    )

    start = time.time()

    cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

    clocker.add_cti(data=pre_cti_image.native, cti=cti)

    negative_pruning_time = time.time() - start

    """
    __Profile No Pruning__
    
    The time to add CTI without pruning
    """
    clocker = ac.Clocker2D(
        parallel_express=parallel_express,
        parallel_roe=parallel_roe,
        parallel_prune_frequency=0,
    )

    start = time.time()

    clocker.add_cti(data=pre_cti_image.native, cti=cti)

    no_pruning_time = time.time() - start

    print(
        f"Times (Pruning / Negative Pruning / No Pruning) = {pruning_time} | {negative_pruning_time} | {no_pruning_time}"
    )

    """
    Finished.
    """
