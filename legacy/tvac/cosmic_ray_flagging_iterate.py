"""
Modeling: Charge Injection Uniform
==================================

In this script, we will fit charge injection imaging to calibrate CTI, where:

 - The CTI model consists of two parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 - The `ImagingCI` is simulated with uniform charge injection lines and no cosmic rays.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from astropy.io import fits
import numpy as np
from os import path
import autocti as ac
import autocti.plot as aplt

cr_threshold = 4.0

"""
__Dataset__
The paths pointing to the dataset we will use for cti modeling.
"""
dataset_name = "EUC_VIS_EXP_000002-00-0.fits_sa_xt_bs_nl.003"

dataset_path = path.join("tvac", "dataset", dataset_name)

imaging_ci_list = []

data_hdulist = fits.open(path.join(dataset_path, dataset_name))

sci_header = data_hdulist[0].header
data_header = data_hdulist[1].header

gain = data_header["gain"]
data_electrons = data_hdulist[1].data * gain

ccd_id = data_header["CCDID"]
date_obs = sci_header["DATE"]

data_header["CI_IJON"] = sci_header["CI_IJON"]
data_header["CI_IJOFF"] = sci_header["CI_IJOFF"]
data_header["CI_VSTAR"] = sci_header["CI_VSTAR"]
data_header["CI_VEND"] = sci_header["CI_VEND"]

data = ac.euclid.Array2DEuclid.from_fits_header(
    array=data_electrons.astype("float"), ext_header=data_header
)

layout = ac.Layout2DCI.from_euclid_fits_header(
    ext_header=data_header,
)

"""
__Correct Serial CTI (So charge injection FPRs are unmixed)__
"""
clocker_2d = ac.Clocker2D(
    parallel_express=5,
    parallel_roe=ac.ROEChargeInjection(),
    serial_express=5,
    serial_prune_n_electrons=1e-7,
    serial_prune_frequency=10,
)

### INSERT PARALLEL / SERIAL HERE ###

data_corrected = data

# cti_2d = ac.CTI2D()
#
# data_corrected = clocker_2d.remove_cti(
#     data=data,
#     cti=cti_2d
# )

injection_on = sci_header["CI_IJON"]

iterations = 3

for i in range(iterations):

    """
    __Charge Injection Estimate__
    """
    injection_norm_list = layout.extract.parallel_fpr.median_list_from(
        array=data_corrected, pixels=(injection_on - 20, injection_on)
    )

    pre_cti_data = layout.pre_cti_data_non_uniform_from(
        injection_norm_list=injection_norm_list,
        pixel_scales=data.pixel_scales,
    )

    """
    __Estimate Noise Map__
    """
    injection_std_list = layout.extract.parallel_fpr.std_list_from(
        array=data, pixels=(injection_on - 20, injection_on)
    )

    noise_map = layout.noise_map_non_uniform_from(
        injection_std_list=injection_std_list,
        pixel_scales=data.pixel_scales,
        read_noise=4.0,
    )

    """
    __Charge Injection Add CTI (So EPER / FPR subtract correctly from data for flagging).
    """
    # pre_cti_data_with_cti = clocker_2d.add_cti(
    #     data=pre_cti_data,
    #     cti=cti_2d
    # )

    pre_cti_data_with_cti = pre_cti_data

    """
    __Charge Injection Subtract__
    """
    data_subtracted = data.native - pre_cti_data_with_cti.native

    """
    __Cosmic Ray Flagging__
    """

    cosmic_ray_mask = data_subtracted.native > cr_threshold * noise_map.native
    cosmic_ray_mask = ac.Mask2D(mask=cosmic_ray_mask, pixel_scales=data.pixel_scales)

    data_corrected = data_corrected.apply_mask(mask=cosmic_ray_mask)

print(layout.region_list)

pre_cti_data.output_to_fits(
    file_path=path.join(dataset_path, "pre_cti_data.fits"), overwrite=True
)

"""
__Data__
"""
mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=np.max(pre_cti_data), vmin=0.0),
    output=aplt.Output(
        path=path.join("tvac", "dataset", dataset_name), filename="data", format=["png"]
    ),
)

array_2d_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()


"""
__Data (Zoom)__
"""
mat_plot_2d = aplt.MatPlot2D(
    axis=aplt.Axis(extent=[0.0, 13.0, 0.0, 13.0]),
    cmap=aplt.Cmap(vmax=np.max(pre_cti_data), vmin=0.0),
    output=aplt.Output(
        path=path.join("tvac", "dataset", dataset_name),
        filename="data_zoom",
        format=["png"],
    ),
)

array_2d_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()

"""
__Pre CTI Data__
"""
mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=np.max(pre_cti_data), vmin=0.0),
    output=aplt.Output(
        path=path.join("tvac", "dataset", dataset_name),
        filename="pre_cti_data",
        format=["png"],
    ),
)

array_2d_plotter = aplt.Array2DPlotter(array=pre_cti_data, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()

"""
__Pre CTI Data (Zoom)__
"""
mat_plot_2d = aplt.MatPlot2D(
    axis=aplt.Axis(extent=[0.0, 13.0, 0.0, 13.0]),
    cmap=aplt.Cmap(vmax=np.max(pre_cti_data), vmin=0.0),
    output=aplt.Output(
        path=path.join("tvac", "dataset", dataset_name),
        filename="pre_cti_data_zoom",
        format=["png"],
    ),
)

array_2d_plotter = aplt.Array2DPlotter(array=pre_cti_data, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()


"""
__Pre CTI Residual Map__
"""
pre_cti_residual_map = data - pre_cti_data

mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=10.0, vmin=-10.0),
    output=aplt.Output(
        path=dataset_path, filename="pre_cti_residual_map", format="png"
    ),
)
array_2d_plotter = aplt.Array2DPlotter(
    array=pre_cti_residual_map, mat_plot_2d=mat_plot_2d
)
array_2d_plotter.figure_2d()


"""
__Pre CTI Residual Map (Zoom)__
"""
pre_cti_residual_map = data - pre_cti_data

mat_plot_2d = aplt.MatPlot2D(
    axis=aplt.Axis(extent=[0.0, 13.0, 0.0, 13.0]),
    cmap=aplt.Cmap(vmax=10.0, vmin=-10.0),
    output=aplt.Output(
        path=dataset_path, filename="pre_cti_residual_map_zoom", format="png"
    ),
)
array_2d_plotter = aplt.Array2DPlotter(
    array=pre_cti_residual_map, mat_plot_2d=mat_plot_2d
)
array_2d_plotter.figure_2d()

"""
__Estimate Noise Map__
"""
injection_std_list = layout.extract.parallel_fpr.std_list_from(
    array=data, pixels=(injection_on - 20, injection_on)
)

noise_map = layout.noise_map_non_uniform_from(
    injection_std_list=injection_std_list,
    pixel_scales=data.pixel_scales,
    read_noise=4.0,
)

noise_map.output_to_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), overwrite=True
)

"""
__Noise Map__
"""
mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=np.max(noise_map), vmin=0.0),
    output=aplt.Output(
        path=path.join("tvac", "dataset", dataset_name),
        filename="noise_map",
        format=["png"],
    ),
)

array_2d_plotter = aplt.Array2DPlotter(array=noise_map, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()

"""
__Noise Map (Zoom)__
"""
mat_plot_2d = aplt.MatPlot2D(
    axis=aplt.Axis(extent=[0.0, 13.0, 0.0, 13.0]),
    cmap=aplt.Cmap(vmax=np.max(noise_map), vmin=0.0),
    output=aplt.Output(
        path=path.join("tvac", "dataset", dataset_name),
        filename="noise_map_zoom",
        format=["png"],
    ),
)

array_2d_plotter = aplt.Array2DPlotter(array=noise_map, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()

"""
__Cosmic Rays__
"""
mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=1.0, vmin=0.0),
    output=aplt.Output(path=dataset_path, filename="cr_mask", format="png"),
)

cosmic_ray_mask_plot = ac.Array2D.no_mask(
    values=cosmic_ray_mask,
    pixel_scales=data.pixel_scales,
).native
array_2d_plotter = aplt.Array2DPlotter(
    array=cosmic_ray_mask_plot, mat_plot_2d=mat_plot_2d
)
array_2d_plotter.figure_2d()
