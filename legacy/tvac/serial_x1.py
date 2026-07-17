"""
Modeling: Charge Injection Uniform
==================================

In this script, we will fit charge injection imaging to calibrate CTI, where:

 - The CTI model consists of two serial `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 - The `ImagingCI` is simulated with uniform charge injection lines and no cosmic rays.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from astropy.io import fits
from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__

The paths pointing to the dataset we will use for cti modeling.
"""
dataset_name = "EUC_VIS_EXP_000000-00-0.fits_sa_xt_bs_nl.003"

dataset_path = path.join("tvac", "dataset", dataset_name)

imaging_ci_list = []

data_hdulist = fits.open(dataset_path)

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

image_ci = ac.euclid.Array2DEuclid.from_fits_header(
    array=data_electrons.astype("float"), ext_header=data_header
)

layout_ci = ac.Layout2DCI.from_euclid_fits_header(
    ext_header=data_header,
)

print(layout_ci.region_list)

injection_on = sci_header["CI_IJON"]

injection_norm_list = layout_ci.extract.parallel_fpr.median_list_from(
    array=image_ci, pixels=(injection_on - 20, injection_on)
)

pre_cti_data = layout_ci.pre_cti_data_non_uniform_from(
    injection_norm_list=injection_norm_list, pixel_scales=image_ci.pixel_scales
)

ci_noise_map = ac.Array2D.full(
    fill_value=4.0,
    shape_native=image_ci.shape_native,
    pixel_scales=image_ci.pixel_scales,
)

imaging_ci = ac.ImagingCI(
    image=image_ci,
    noise_map=ci_noise_map,
    pre_cti_data=pre_cti_data,
    layout=layout_ci,
)

imaging_ci_list.append(imaging_ci)

"""
 __Example: Modeling__

 To fit a CTI model to a dataset, we must perform CTI modeling, which uses a non-linear search algorithm to fit many
 different CTI models to the dataset.

 Model-fitting is handled by our project **PyAutoFit**, a probablistic programming language for non-linear model
 fitting. The setting up on configuration files is performed by our project **PyAutoConf**. We'll need to import
 both to perform the model-fit.

 In this script, we will fit charge injection imaging which has been subjected to CTI, where:

  - The CTI model consists of two serial `Trap` species.
  - The `CCD` volume fill parameterization is a simple form with just a *well_fill_beta* parameter.
  - The `CIImaging` is simulated with uniform charge injection lines and no cosmic rays.

 The *Clocker* models the CCD read-out, including CTI. 

 For serial clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
 """
clocker = ac.Clocker2D(serial_express=5)

"""
__Model__

We compose our lens model using `Trap` and `CCD` objects, which are what add CTI to our images during clocking and 
read out. In this example our CTI model is:

 - Two serial `TrapInstantCapture`'s which capture electrins during clokcing intant in the serial direction. 
 - A simple `CCD` volume beta parametrization.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""

serial_trap_list = [af.Model(ac.TrapInstantCapture)]
serial_ccd = af.Model(ac.CCDPhase)
serial_ccd.well_notch_depth = 0.0
serial_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(ac.CTI2D, serial_trap_list=serial_trap_list, serial_ccd=serial_ccd)
)

"""
__Search__

The lens model is fitted to the data using a `NonLinearSearch`. In this example, we use the
nested sampling algorithm MultiNest with 50 live points.

The script 'autocti_workspace/examples/model/customize/non_linear_searches.py' gives a description of the types of
non-linear searches that can be used with **PyAutoCTI**. If you do not know what a non-linear search is or how it 
operates, checkout chapters 1 and 2 of the HowToCTI lecture series.
"""
search = af.DynestyStatic(
    name="serial[x1]",
    n_live_points=50,
    vol_dec=0.5,
    vol_check=2.0,
    unique_tag=dataset_name,
)

"""
__Settings__

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 2000 columns) to just 50 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.

We also mask the FPR of the data during the model-fit.
"""
mask_ci = ac.Mask2D.all_false(
    shape_native=imaging_ci_list[0].shape_native,
    pixel_scales=imaging_ci_list[0].pixel_scales,
)

mask_ci = ac.Mask2D.masked_fpr_and_eper_from(
    mask=mask_ci,
    layout=imaging_ci_list[0].layout,
    settings=ac.SettingsMask2D(serial_fpr_pixels=(0, 2048)),
    pixel_scales=imaging_ci_list[0].pixel_scales,
)

imaging_ci_trimmed_list = [
    imaging_ci.apply_mask(mask=mask_ci) for imaging_ci in imaging_ci_list
]

imaging_ci_trimmed_list = [
    imaging_ci.apply_settings(settings=ac.SettingsImagingCI(serial_pixels=(20, 40)))
    for imaging_ci in imaging_ci_trimmed_list
]

"""
__Analysis__

The `AnalysisImagingCI` object defines the `log_likelihood_function` used by the non-linear search to fit the 
model to  the `ImagingCI`dataset.
"""
analysis_list = [
    ac.AnalysisImagingCI(dataset=imaging_ci, clocker=clocker)
    for imaging_ci in imaging_ci_trimmed_list
]

analysis = sum(analysis_list)

"""
We can now begin the fit by passing the dataset and mask to the phase, which will use the non-linear search to fit
the model to the data. 

The fit outputs visualization on-the-fly, so checkout the path 
'/path/to/autolens_workspace/output/examples/phase__lens_sie__source_sersic' to see how your fit is doing!
"""
result_list = search.fit(model=model, analysis=analysis)
