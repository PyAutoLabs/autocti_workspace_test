"""
Correction: Charge Injection Uniform
====================================

In this script, we correct CTI from charge injection imaging using a known CTI model.

Whilst correcting CTI calibration data is not something one would commonly do, this script is here to illustrate
the API for performing CTI correction.

The correction of CTI calibration data can also be used as a diagnostic for the quality of the CTI model that is
calibrated.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__

The paths pointing to the dataset we will use for cti modeling.
"""
dataset_type = "validation"
dataset_name = "parallel_x1"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (2000, 100)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 80))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 80, 100))

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
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
imaging_ci_list = [
    ac.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        layout=layout,
        pixel_scales=0.1,
    )
    for layout, norm in zip(layout_list, norm_list)
]

"""
Lets plot the first `ImagingCI`.
"""
imaging_ci_plotter = aplt.ImagingCIPlotter(dataset=imaging_ci_list[0])
imaging_ci_plotter.subplot_imaging_ci()

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker_2d = ac.Clocker2D(parallel_express=2)

"""
__Model__

We now compose the CTI model we will use to correct CTI from the data.

In this example, the true CTI model used to simulate the data is specified below. The `results` and `database` 
packages have tutorials showing how to directly use the results of a CTI calibration.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=1.0, release_timescale=5.0)
parallel_trap_list = [parallel_trap_0]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

"""
__Correction__

We use the CTI model and clocker to perform the CTI correction.
"""
data_corrected_2d_list = [
    clocker_2d.remove_cti(data=imaging_ci.data, cti=cti)
    for imaging_ci in imaging_ci_list
]

"""
__Corrected Imaging CI__


"""
imaging_ci_list = [
    ac.ImagingCI(
        image=data_corrected_2d,
        noise_map=imaging_ci.noise_map,
        pre_cti_data=imaging_ci.pre_cti_data,
        layout=imaging_ci.layout,
    )
    for data_corrected_2d, imaging_ci in zip(data_corrected_2d_list, imaging_ci_list)
]

"""
__Dataset 1D__

Create a 1D dataset of the parallel EPERs, by binning up over the parallel overscan.
"""
dataset_1d = imaging_ci_list[0].layout.extract.parallel_overscan.dataset_1d_from(
    dataset_2d=imaging_ci_list[0], pixels=(-100, 20)
)

mask_1d = ac.Mask1D.all_false(
    shape_slim=dataset_1d.shape_slim, pixel_scales=dataset_1d.pixel_scales
)


mask_1d = ac.Mask1D.masked_fpr_and_eper_from(
    mask=mask_1d,
    layout=dataset_1d.layout,
    settings=ac.SettingsMask1D(fpr_pixels=(0, 100)),
    pixel_scales=dataset_1d.pixel_scales,
)

dataset_1d = dataset_1d.apply_mask(mask=mask_1d)

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 
"""
clocker_1d = ac.Clocker1D(express=2)

"""
__Model__

We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the cti 
1D data. In this example we fit a CTI model with:

 - One `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
 [2 parameters].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""
trap = af.Model(ac.TrapInstantCapture)

# trap.release_timescale = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
trap_list = [trap]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

"""
__Search__

The CTI model is fitted to the data using a `NonLinearSearch`. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).

The script 'autocti_workspace/examples/modeling/customize/non_linear_searches.py' gives a description of the types of
non-linear searches that can be used with **PyAutoCTI**. If you do not know what a `NonLinearSearch` is or how it 
operates, checkout chapter 2 of the HowToCTI lecture series.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autocti_workspace/output/dataset_1d/species[x2]`.
"""
search = af.DynestyStatic(
    name="cti_valid_parallel_x1", nlive=50, sample="rwalk", walks=10
)

"""
__Analysis__

The `AnalysisDataset1D` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `Dataset1D`dataset.
"""
analysis = ac.AnalysisDataset1D(dataset=dataset_1d, clocker=clocker_1d)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autocti_workspace/output/dataset_1d/species[x2]` for live outputs of the results of the fit, 
including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

instance = result.max_log_likelihood_instance

print(f"Inferred Density: = {instance.cti.trap_list[0].density}")
print(f"Inferred Delta Ell: = {instance.cti.delta_ellipticity}")

print(f"Likelihood ML: {analysis.log_likelihood_function(instance=instance)}")

instance.cti.trap_list[0].density = 0.0

print(f"Likelihood Density 0:  {analysis.log_likelihood_function(instance=instance)}")

instance.cti.trap_list[0].density = 0.5

print(f"Likelihood Density 0.5: {analysis.log_likelihood_function(instance=instance)}")

instance.cti.trap_list[0].density = 5.0

print(f"Likelihood Density 5.0: {analysis.log_likelihood_function(instance=instance)}")
"""
Finished.
"""
