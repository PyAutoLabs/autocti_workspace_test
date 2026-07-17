"""
Simulator: Temporal
===================

This script simulates multiple charge injection imaging datasets with CTI, where:

 - Parallel CTI is added to the image using a 1 `Trap` species model.
 - The volume filling behaviour in the parallel direction using the `CCD` class.
 - The `ImagingCI` is simulated with uniform charge injection lines and no cosmic rays.

The multiple datasets are representative of CTI calibration data taken over the course of a space mission. Therefore,
the density of traps increases with time.

The charge injection data is a small cut-out of 30 x 30 pixels, to make CTI calibration run fast for the overview
script that this data is used for.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Dataset Paths__

The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/noise_map.fits'.
 - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/pre_cti_data.fits'.
"""
dataset_label = "temporal"
dataset_type = "parallel_x1"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/overview/uniform'
"""
dataset_path = path.join(dataset_label, "dataset", dataset_type)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (30, 30)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((25, 30, 1, 29))
serial_prescan = ac.Region2D((0, 30, 0, 1))
serial_overscan = ac.Region2D((0, 25, 29, 30))

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
regions_list = [(0, 5, serial_prescan[3], serial_overscan[2])]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of charge injection images that are simulated.
"""
total_ci_images = len(norm_list)

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
layout_list = [
    ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=regions_list,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for i in range(total_ci_images)
]

"""
__Clocker__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker2D(parallel_express=2, parallel_roe=ac.ROEChargeInjection())

"""
__Base CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 1 `TrapInstantCapture` species in the parallel and serial directions, which captures electrons during clocking 
 instantly and release them according to an exponential probability distribution defined by a single release times.

 - A simple CCD volume beta parametrization for parallel and serial clocking separately.
 
__Temporal CTI Model__

We will create 10 realisations of the above model, corresponding to CTI calibration data taken at ten equally spaced
intervals of the space mission.

The density of the trap species for each dataset is computed via a linear relation between time and density, where:

 y = mx + c
 
 x = time
 m = density evolution
 c = density at mission start
 y = density at a given time
"""

density_evolution = 0.05
density_start = 0.01

time_list = range(0, 2)

for time in time_list:

    """
    __Density at Time__

    Compute the density of the trap from the linear relation defining its time evolution.
    """
    density = float((density_evolution * time) + density_start)

    parallel_trap_0 = ac.TrapInstantCapture(density=density, release_timescale=5.0)
    parallel_trap_list = [parallel_trap_0]

    parallel_ccd = ac.CCDPhase(
        well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0
    )

    cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

    """
    __Simulate__
    
    To simulate charge injection imaging, we pass the charge injection pattern to a `SimulatorImagingCI`, which adds CTI 
    via arCTIc and read-noise to the data.
    
    This creates instances of the `ImagingCI` class, which include the images, noise-maps and pre_cti_data images.
    """
    simulator_list = [
        ac.SimulatorImagingCI(read_noise=4.0, pixel_scales=0.1, norm=norm)
        for norm in norm_list
    ]

    """
    We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
    and before passing each image to arCTIc does the following:
    
     - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
     - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
     - Includes information on the different scan regions of the image, such as the serial prescan and serial overscan.
    """
    imaging_ci_list = [
        simulator.via_layout_from(clocker=clocker, layout=layout_ci, cti=cti)
        for layout_ci, simulator in zip(layout_list, simulator_list)
    ]

    """
    __Output__
    
    We output each simulated dataset to a folder based on its number of times.
    
    Output a subplot of the simulated dataset to the dataset path as .png files.
    """
    dataset_time = f"time_{time}"
    dataset_output_path = path.join(dataset_path, dataset_time)

    mat_plot_2d = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_output_path, format="png")
    )

    for imaging_ci, norm in zip(imaging_ci_list, norm_list):
        output = aplt.Output(
            path=path.join(dataset_output_path, f"norm_{int(norm)}"),
            filename="imaging_ci",
            format="png",
        )

        mat_plot_2d = aplt.MatPlot2D(output=output)

        imaging_ci_plotter = aplt.ImagingCIPlotter(
            dataset=imaging_ci, mat_plot_2d=mat_plot_2d
        )
        imaging_ci_plotter.subplot_imaging_ci()

    """
    Output plots of the EPER and FPR's binned up in 1D, so that electron capture and trailing can be
    seen clearly.
    """
    for imaging_ci, norm in zip(imaging_ci_list, norm_list):
        output = aplt.Output(
            path=path.join(dataset_output_path, f"norm_{int(norm)}", "binned_1d"),
            format="png",
        )

        mat_plot_1d = aplt.MatPlot1D(output=output)

        imaging_ci_plotter = aplt.ImagingCIPlotter(
            dataset=imaging_ci, mat_plot_1d=mat_plot_1d
        )
        imaging_ci_plotter.figures_1d_of_region(region="parallel_fpr", image=True)
        imaging_ci_plotter.figures_1d_of_region(region="parallel_eper", image=True)

    """
    Output the image, noise-map and pre cti image of the charge injection dataset to .fits files.
    """
    [
        imaging_ci.output_to_fits(
            image_path=path.join(dataset_output_path, f"data_{int(norm)}.fits"),
            noise_map_path=path.join(
                dataset_output_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_output_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            overwrite=True,
        )
        for imaging_ci, norm in zip(imaging_ci_list, norm_list)
    ]

    """
    Save the `TrapInstantCapture` in the dataset folder as a .json file, ensuring the true densities
    are safely stored and available to check how the dataset was simulated in the future. 

    This can be loaded via the method `TrapInstantCapture.from_json`.
    """
    cti.output_to_json(file_path=path.join(dataset_output_path, "cti.json"))
    clocker.output_to_json(file_path=path.join(dataset_output_path, "clocker.json"))

"""
Finished.
"""
