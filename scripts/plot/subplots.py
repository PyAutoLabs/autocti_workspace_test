"""
Integration: plot function API
==============================

Drives the main `autocti.plot` function surface — dataset and fit subplots, region figures,
combined lists and the binned-data diagnostic — writing every figure to disk, for both a 1D
dataset and a 2D charge injection dataset.
"""

import shutil
from os import path

import autocti as ac
import autocti.plot as aplt

output_root = path.join("output", "plot_integration")

if path.exists(output_root):
    shutil.rmtree(output_root)

output_path_1d = path.join(output_root, "dataset_1d")
output_path_2d = path.join(output_root, "imaging_ci")

"""
__Dataset1D__
"""
layout_1d = ac.Layout1D(
    shape_1d=(50,),
    region_list=[(5, 25)],
    prescan=ac.Region1D(region=(0, 5)),
    overscan=ac.Region1D(region=(45, 50)),
)

clocker_1d = ac.Clocker1D(express=5, roe=ac.ROEChargeInjection())
cti_1d = ac.CTI1D(
    trap_list=[ac.TrapInstantCapture(density=10.0, release_timescale=5.0)],
    ccd=ac.CCDPhase(well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0),
)

simulator_1d = ac.SimulatorDataset1D(read_noise=0.01, pixel_scales=0.1, norm=100.0)
dataset_1d = simulator_1d.via_layout_from(
    clocker=clocker_1d, layout=layout_1d, cti=cti_1d
)

aplt.figure_dataset_1d_data(
    dataset=dataset_1d, output_path=output_path_1d, output_format="png"
)
aplt.figure_dataset_1d_data(
    dataset=dataset_1d, region="eper", logy=True, output_path=output_path_1d, output_format="png"
)
aplt.subplot_dataset_1d(dataset=dataset_1d, output_path=output_path_1d, output_format="png")
aplt.subplot_dataset_1d(
    dataset=dataset_1d, region="fpr", output_path=output_path_1d, output_format="png"
)
aplt.subplot_dataset_1d_list(
    dataset_list=[dataset_1d, dataset_1d], output_path=output_path_1d, output_format="png"
)

fit_1d = ac.FitDataset1D(
    dataset=dataset_1d.apply_mask(
        mask=ac.Mask1D.all_false(
            shape_slim=dataset_1d.data.shape_slim,
            pixel_scales=dataset_1d.data.pixel_scales,
        )
    ),
    post_cti_data=clocker_1d.add_cti(data=dataset_1d.pre_cti_data, cti=cti_1d),
)

aplt.subplot_fit_dataset_1d(fit=fit_1d, output_path=output_path_1d, output_format="png")
aplt.figure_fit_dataset_1d(
    fit=fit_1d,
    quantity="residual_map",
    region="eper",
    output_path=output_path_1d,
    output_format="png",
)
aplt.fits_fit_dataset_1d(fit=fit_1d, output_path=output_path_1d)

"""
__ImagingCI__
"""
layout_2d = ac.Layout2DCI(
    shape_2d=(30, 30),
    region_list=[(0, 25, 1, 29)],
    parallel_overscan=ac.Region2D((25, 30, 1, 29)),
    serial_prescan=ac.Region2D((0, 30, 0, 1)),
    serial_overscan=ac.Region2D((0, 25, 29, 30)),
)

clocker_2d = ac.Clocker2D(parallel_express=5, parallel_roe=ac.ROEChargeInjection())
cti_2d = ac.CTI2D(
    parallel_trap_list=[ac.TrapInstantCapture(density=10.0, release_timescale=5.0)],
    parallel_ccd=ac.CCDPhase(
        well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0
    ),
)

simulator_2d = ac.SimulatorImagingCI(read_noise=0.01, pixel_scales=0.1, norm=100.0)
dataset_2d = simulator_2d.via_layout_from(
    clocker=clocker_2d, layout=layout_2d, cti=cti_2d
)

aplt.plot_array(
    array=dataset_2d.data,
    title="Data",
    output_path=output_path_2d,
    output_filename="data_2d",
    output_format="png",
)
aplt.subplot_imaging_ci(dataset=dataset_2d, output_path=output_path_2d, output_format="png")
aplt.subplot_imaging_ci_region(
    dataset=dataset_2d, region="parallel_eper", output_path=output_path_2d, output_format="png"
)
aplt.figure_imaging_ci_data_region(
    dataset=dataset_2d,
    region="parallel_fpr",
    output_path=output_path_2d,
    output_format="png",
)
aplt.subplot_imaging_ci_data_binned(
    dataset=dataset_2d, output_path=output_path_2d, output_format="png"
)

fit_2d = ac.FitImagingCI(
    dataset=dataset_2d.apply_mask(
        mask=ac.Mask2D.all_false(
            shape_native=dataset_2d.shape_native, pixel_scales=dataset_2d.pixel_scales
        )
    ),
    post_cti_data=clocker_2d.add_cti(data=dataset_2d.pre_cti_data, cti=cti_2d),
)

aplt.subplot_fit_ci(fit=fit_2d, output_path=output_path_2d, output_format="png")
aplt.subplot_fit_ci_region(
    fit=fit_2d, region="parallel_eper", output_path=output_path_2d, output_format="png"
)
aplt.fits_fit_ci(fit=fit_2d, output_path=output_path_2d)

"""
__Verify__
"""
import glob

n_png_1d = len(glob.glob(path.join(output_path_1d, "*.png")))
n_png_2d = len(glob.glob(path.join(output_path_2d, "*.png")))
n_fits = len(glob.glob(path.join(output_root, "*", "*.fits")))

assert n_png_1d >= 7, f"expected >= 7 dataset_1d pngs, found {n_png_1d}"
assert n_png_2d >= 7, f"expected >= 7 imaging_ci pngs, found {n_png_2d}"
assert n_fits >= 2, f"expected >= 2 fits files, found {n_fits}"

print(f"plot integration: OK ({n_png_1d}+{n_png_2d} pngs, {n_fits} fits)")
