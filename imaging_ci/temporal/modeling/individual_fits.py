"""
Temporal: Individual Fits
=========================

In this script, we will fit multiple charge injection imaging to calibrate CTI, where:

 - The CTI model consists of one parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 - The `ImagingCI` is simulated with uniform charge injection lines and no cosmic rays.

The multiple datasets are representative of CTI calibration data taken over the course of a space mission. Therefore,
the model-fitting aims to determine the increase in the density of traps with time.

This script fits each dataset one-by-one and uses the results post-analysis to determine the density evolution
parameters. Other scripts perform more detailed analyses that use more advanced statistical methods to provide a
better estimate.

The charge injection data is a small cut-out of 30 x 30 pixels, to make CTI calibration run fast for the overview
script that this data is used for.
"""
import numpy as np
from os import path
import autofit as af
import autocti as ac

"""
__Dataset__

The paths pointing to the dataset we will use for cti modeling.
"""
dataset_label = "temporal"
dataset_type = "parallel_x1"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/overview/uniform'
"""
dataset_path = path.join("dataset", "imaging_ci", dataset_label, dataset_type)

"""
__Layout__

The 2D shape of the images.
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
We now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.

We load and fit each dataset, accquried on different months, one-by-one. We do this in a for loop to avoid loading 
everything into memory.
"""
month_list = list(np.linspace(0.0, 36.0, 10))

for month in month_list:

    dataset_months = f"month_{month}"
    dataset_month_path = path.join(dataset_path, dataset_months)

    imaging_ci_list = [
        ac.ImagingCI.from_fits(
            image_path=path.join(dataset_month_path, f"data_{int(norm)}.fits"),
            noise_map_path=path.join(
                dataset_month_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_month_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            layout=layout,
            pixel_scales=0.1,
        )
        for layout, norm in zip(layout_list, norm_list)
    ]

    """
    __Mask__
    
    We apply a 2D mask which removes the FPR (e.g. all 5 pixels where the charge injection is performed).
    """
    mask_2d = ac.Mask2D.all_false(
        shape_native=imaging_ci_list[0].shape_native,
        pixel_scales=imaging_ci_list[0].pixel_scales,
    )

    mask_2d = ac.Mask2D.masked_fpr_and_eper_from(
        mask=mask_2d,
        layout=imaging_ci_list[0].layout,
        settings=ac.SettingsMask2D(parallel_fpr_pixels=(0, 5)),
        pixel_scales=imaging_ci_list[0].pixel_scales,
    )

    imaging_ci_list = [
        imaging_ci.apply_mask(mask=mask_2d) for imaging_ci in imaging_ci_list
    ]

    """
    __Clocking__
    
    The `Clocker` models the CCD read-out, including CTI. 
    
    For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
    """
    clocker = ac.Clocker2D(parallel_express=2, parallel_roe=ac.ROEChargeInjection())

    """
    __Model__
    
    We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the 
    charge  injection data. In this example we fit a CTI model with:
    
     - One parallel `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
     [2 parameters].
    
     - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].
    
    The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.
    """
    parallel_trap_0 = af.Model(ac.TrapInstantCapture)
    parallel_traps = [parallel_trap_0]
    parallel_ccd = af.Model(ac.CCDPhase)
    parallel_ccd.well_notch_depth = 0.0
    parallel_ccd.full_well_depth = 200000.0

    model = af.Collection(
        cti=af.Model(
            ac.CTI2D, parallel_trap_list=[parallel_trap_0], parallel_ccd=parallel_ccd
        )
    )

    """
    __Search__
    
    The CTI model is fitted to the data using a `NonLinearSearch`. In this example, we use the
    nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).
    """
    search = af.DynestyStatic(
        path_prefix=path.join("imaging_ci", dataset_label, dataset_months),
        name="parallel[x1]",
        nlive=50,
    )

    """
    __Analysis__
    
    The `AnalysisImagingCI` object defines the `log_likelihood_function` used by the non-linear search to fit the 
    model to the `ImagingCI` dataset.
    """
    analysis_list = [
        ac.AnalysisImagingCI(dataset=imaging_ci, clocker=clocker)
        for imaging_ci in imaging_ci_list
    ]
    analysis = sum(analysis_list)
    analysis.n_cores = 1

    """
    __Info__

    An `info_dict` containing additional information about the data canbe passed into the model-fit, which can be
    loaded post-analysis for estimate the time-evolution of the CTI model.

    For each fit we pass in an info dictionary containing the number of months into the space mission the dataset 
    was observed.
    """
    info_dict = {"months": month}

    """
    __Model-Fit__
    
    We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
    search to find which models fit the data with the highest likelihood.
    """
    result_list = search.fit(model=model, analysis=analysis, info=info_dict)
