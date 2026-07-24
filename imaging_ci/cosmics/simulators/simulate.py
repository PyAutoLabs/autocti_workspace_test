"""
Simulator: Non-Uniform Charge Injection With Cosmic Rays
========================================================

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallle direction using the `CCD` class.
"""
from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Path__

Grab the relative path to this file for loading cosmic ray files and outputting the cosmic ray map.
"""
dir_path = path.dirname(path.realpath(__file__))

"""
__Dataset Paths__

The 'dataset_type' describes the type of data being simulated (in this case, imaging data). They define the folder 
the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_type/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_type/noise_map.fits'.
 - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_type/pre_cti_data.fits'.
"""
dataset_type = "imaging_ci_non_uniform"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/uniform_cosmic_rays/parallel_x1'
"""
dataset_path = path.join(dir_path, "dataset", dataset_type)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (2086, 2128)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, shape_native[1] - 5))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, shape_native[1] - 5, shape_native[1]))

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
regions_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [100, 5000, 25000, 200000]

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
# layout_list = [
#     ac.Layout2DCI(
#         shape_2d=shape_native,
#         region_list=regions_list,
#         normalization=normalization,
#         parallel_overscan=parallel_overscan,
#         serial_prescan=serial_prescan,
#         serial_overscan=serial_overscan,
#     )
#     for normalization in normalization_list
# ]

column_sigma_list = [0.1 * norm for norm in norm_list]
row_slope_list = [0.0] * len(norm_list)

layout_list = [
    ac.Layout2DCINonUniform(
        shape_2d=shape_native,
        region_list=regions_list,
        norm=norm,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
        column_sigma=column_sigma,
        row_slope=row_slope,
        maximum_norm=200000.0,
    )
    for (norm, column_sigma, row_slope) in zip(
        norm_list, column_sigma_list, row_slope_list
    )
]

"""
__Clocker__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker2D(parallel_express=2, parallel_roe=ac.ROEChargeInjection())

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 1 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
"""
parallel_trap = ac.TrapInstantCapture(density=0.0, release_timescale=4.0)
parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

"""
__Simulate__

To simulate charge injection imaging, we pass the charge injection pattern to a `SimulatorImagingCI`, which adds CTI 
via arCTIc and read-noise to the data.

This creates instances of the `ImagingCI` class, which include the images, noise-maps and pre_cti_data images.
"""
simulator = ac.SimulatorImagingCI(read_noise=4.0, pixel_scales=0.1)

"""
We also need to simulate the cosmic ray map, which we pass to the imaging simulator above. These cosmic rays will 
then be added to our ci pre-cti image in the simulate function below, and subject to CTI according to the CTI model.

This uses the `SimulatorCosmicRayMap` to simulator cosmic rays via a random monte carlo process. The settings of the
simulator can be customized via input `.fits` files, but we simply use the defaults supplied with **PyAutoCTI**.
"""
simulator_cosmic_ray_map = ac.SimulatorCosmicRayMap.defaults(
    shape_native=shape_native,
    flux_scaling=1.0,
    pixel_scale=simulator.pixel_scales,
    seed=1,
)

"""
We now iterate over every normalization to create the corresponding cosmic ray maps.

To ensure cosmic rays are not simulated above the CCD full well depth, the `limit` parameter caps all cosmic rays to
this value.
"""
cosmic_ray_map_list = list(
    map(
        lambda i: simulator_cosmic_ray_map.cosmic_ray_map_from(
            limit=parallel_ccd.full_well_depth
        ),
        range(len(norm_list)),
    )
)

"""
We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
and before passing each image to arCTIc does the following:

 - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
 - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
 - Includes information on the different scan regions of the image, such as the serial prescan and serial overscan.
"""
imaging_ci_list = [
    simulator.via_layout_from(
        layout=layout,
        clocker=clocker,
        parallel_trap_list=[parallel_trap],
        parallel_ccd=parallel_ccd,
        cosmic_ray_map=cosmic_ray_map,
    )
    for layout, cosmic_ray_map in zip(layout_list, cosmic_ray_map_list)
]

"""
__Output__

Output a subplot of the simulated dataset to the dataset path as .png files.
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

imaging_ci_plotter = aplt.ImagingCIPlotter(
    dataset=imaging_ci_list[0], mat_plot_2d=mat_plot_2d
)
imaging_ci_plotter.subplot_imaging_ci()

"""
Output the image, noise-map and pre cti image of the charge injection dataset to .fits files.
"""
[
    imaging_ci.output_to_fits(
        image_path=path.join(dataset_path, f"norm_{int(norm)}", "image.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        cosmic_ray_map_path=path.join(dataset_path, f"cosmic_ray_map_{int(norm)}.fits"),
        overwrite=True,
    )
    for imaging_ci in imaging_ci_list
]

"""
Finished.
"""
