"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallle direction using the `CCD` class.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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
dataset_type = "validation"
dataset_name = "no_cti"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/uniform/parallel_x2'
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (2000, 2000)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 19980))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 19980, 20000))

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
regions_list = [(0, 1980, serial_prescan[3], serial_overscan[2])]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [200]

"""
The total number of charge injection images that are simulated.
"""
total_images = len(norm_list)

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
    for i in range(total_images)
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
We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
and before passing each image to arCTIc does the following:

 - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
 - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
 - Includes information on the different scan regions of the image, such as the serial prescan and serial overscan.
"""
imaging_ci_list = [
    simulator.via_layout_from(clocker=None, layout=layout_ci, cti=None)
    for layout_ci, simulator in zip(layout_list, simulator_list)
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
        overwrite=True,
    )
    for imaging_ci, norm in zip(imaging_ci_list, norm_list)
]

"""
Finished.
"""
