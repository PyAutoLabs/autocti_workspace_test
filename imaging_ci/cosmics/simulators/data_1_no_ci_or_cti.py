"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallel direction using the `CCD` class.
"""
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
dataset_name = "no_ci_or_cti"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/uniform/parallel_x2'
"""
dataset_path = path.join("imaging_ci", "cosmics", "dataset", dataset_name)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (2086, 2128)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((2066, 2086, 51, shape_native[1] - 29))
serial_prescan = ac.Region2D((0, 2086, 0, 51))
serial_overscan = ac.Region2D((0, 2066, shape_native[1] - 29, shape_native[1]))

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [1]

"""
The total number of charge injection images that are simulated.
"""
total_ci_images = len(norm_list)

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
layout_2d_list = [
    ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=[(0, 1, 0, 1)],
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for i in range(total_ci_images)
]

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
We also need to simulate the cosmic ray map, which we pass to the imaging simulator above. These cosmic rays will 
then be added to our ci pre-cti image in the simulate function below, and subject to CTI according to the CTI model.

This uses the `SimulatorCosmicRayMap` to simulator cosmic rays via a random monte carlo process. The settings of the
simulator can be customized via input `.fits` files, but we simply use the defaults supplied with **PyAutoCTI**.
"""
simulator_cosmic_ray_map = ac.SimulatorCosmicRayMap.defaults(
    shape_native=shape_native,
    flux_scaling=1.0,
    pixel_scale=simulator_list[0].pixel_scales,
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
            cover_fraction=0.2548, limit=200000
        ),
        range(len(norm_list)),
    )
)

cosmic_ray_map_list[0] = cosmic_ray_map_list[0].native

cosmic_ray_map_list[0][0, 0:-1] = 0.0
cosmic_ray_map_list[0][-1, 0:-1] = 0.0
cosmic_ray_map_list[0][0:-1, 0] = 0.0
cosmic_ray_map_list[0][0:-1, -1] = 0.0

"""
We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
and before passing each image to arCTIc does the following:

 - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
 - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
 - Includes information on the different scan regions of the image, such as the serial prescan and serial overscan.
"""
imaging_ci_list = [
    simulator.via_layout_from(
        clocker=None, layout=layout_ci, cti=None, cosmic_ray_map=cosmic_ray_map
    )
    for layout_ci, simulator, cosmic_ray_map in zip(
        layout_2d_list, simulator_list, cosmic_ray_map_list
    )
]

"""
__Output__

Output subplots of the simulated dataset to the dataset path as .png files.
"""
for imaging_ci, norm in zip(imaging_ci_list, norm_list):

    output = aplt.Output(
        path=path.join(dataset_path, f"norm_{int(norm)}"),
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
        path=path.join(dataset_path, f"norm_{int(norm)}", "binned_1d"), format="png"
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
        image_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        cosmic_ray_map_path=path.join(
            dataset_path, f"norm_{int(norm)}", f"cosmic_ray_map.fits"
        ),
        overwrite=True,
    )
    for imaging_ci, norm in zip(imaging_ci_list, norm_list)
]

"""
Finished.
"""
