"""
Integration: ImagingCI model fit
================================

Simulates a small charge injection image with parallel CTI, fits it with a parallel CTI model
via an `AnalysisFactor` / `FactorGraphModel` model-fit, and inspects the result — the core 2D
charge-injection calibration path, end-to-end.

A single trap species is used so the model has no ordering assertion (which ties at prior
medians under the `PYAUTO_TEST_MODE=2` bypass).
"""

from os import path

import autofit as af
import autocti as ac

"""
__Simulate__
"""
shape_native = (30, 30)

parallel_overscan = ac.Region2D((25, 30, 1, 29))
serial_prescan = ac.Region2D((0, 30, 0, 1))
serial_overscan = ac.Region2D((0, 25, 29, 30))

region_list = [(0, 25, serial_prescan[3], serial_overscan[2])]

norm = 100.0

layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=region_list,
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

clocker = ac.Clocker2D(parallel_express=5, parallel_roe=ac.ROEChargeInjection())

parallel_trap = ac.TrapInstantCapture(density=10.0, release_timescale=5.0)
parallel_ccd = ac.CCDPhase(
    well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0
)
cti = ac.CTI2D(parallel_trap_list=[parallel_trap], parallel_ccd=parallel_ccd)

simulator = ac.SimulatorImagingCI(read_noise=0.01, pixel_scales=0.1, norm=norm)

dataset = simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)

"""
__Model + Fit__
"""
model = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=[af.Model(ac.TrapInstantCapture)],
        parallel_ccd=af.Model(
            ac.CCDPhase, well_notch_depth=0.0, full_well_depth=200000.0
        ),
    )
)

search = af.Nautilus(
    path_prefix=path.join("imaging_ci"),
    name="model_fit",
    unique_tag="integration",
    n_live=75,
)

analysis = ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)

analysis_factor_list = [af.AnalysisFactor(prior_model=model, analysis=analysis)]

factor_graph = af.FactorGraphModel(*analysis_factor_list)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

result = result_list[0]

print("max log likelihood:", result.log_likelihood)
print(result.max_log_likelihood_instance.cti.parallel_trap_list[0].density)

fit = result.max_log_likelihood_fit
print("fit chi squared:", fit.chi_squared)

assert fit.chi_squared is not None

density_result = result.max_log_likelihood_instance.cti.parallel_trap_list[0].density

"""
__Aggregator round-trip__

Load the fit back from an .sqlite database and check the dataset, the fit and the CTI model all
come back, with the CTI identical to the one the search returned.
"""
import os
from pathlib import Path

from autocti import conf
from autocti import with_test_mode_segment

database_file = "imaging_ci_integration.sqlite"

if path.exists(database_file):
    os.remove(database_file)

agg = af.Aggregator.from_database(
    filename=database_file, completed_only=False, top_level_only=True
)
agg.add_directory(
    directory=str(with_test_mode_segment(Path(conf.instance.output_path)) / "imaging_ci")
)

total_datasets = sum(
    len(dataset_sublist)
    for dataset_sublist in ac.agg.ImagingCIAgg(aggregator=agg).dataset_list_gen_from()
)
assert total_datasets >= 1, f"aggregator returned {total_datasets} datasets, expected >= 1"

total_fits = sum(
    len(fit_sublist)
    for fit_sublist in ac.agg.FitImagingCIAgg(aggregator=agg).max_log_likelihood_gen_from()
)
assert total_fits >= 1, f"aggregator returned {total_fits} fits, expected >= 1"

cti_list_list = list(ac.agg.CTIAgg(aggregator=agg).max_log_likelihood_gen_from())
assert len(cti_list_list) >= 1, "aggregator returned no CTI models"

density_database = cti_list_list[0][0].parallel_trap_list[0].density
assert density_database == density_result, (
    f"the database CTI (density {density_database}) differs from the search result "
    f"(density {density_result})"
)

if path.exists(database_file):
    os.remove(database_file)

print("imaging_ci integration: OK")
