"""
Integration: Dataset1D model fit
================================

Simulates two 1D CTI datasets at different injection levels, fits them simultaneously via an
`AnalysisFactor` / `FactorGraphModel` model-fit, and round-trips the results through the
aggregator — the core 1D CTI calibration path, end-to-end.

A single trap species is used so the model has no ordering assertion (which ties at prior
medians under the `PYAUTO_TEST_MODE=2` bypass).
"""

import os
from os import path

import autofit as af
import autocti as ac

"""
__Simulate__
"""
shape_native = (50,)

prescan = ac.Region1D(region=(0, 5))
overscan = ac.Region1D(region=(45, 50))
region_list = [(5, 25)]

norm_list = [100.0, 5000.0]

layout = ac.Layout1D(
    shape_1d=shape_native,
    region_list=region_list,
    prescan=prescan,
    overscan=overscan,
)

clocker = ac.Clocker1D(express=5, roe=ac.ROEChargeInjection())

trap = ac.TrapInstantCapture(density=10.0, release_timescale=5.0)
ccd = ac.CCDPhase(well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0)
cti = ac.CTI1D(trap_list=[trap], ccd=ccd)

dataset_list = []

for norm in norm_list:
    simulator = ac.SimulatorDataset1D(
        pixel_scales=0.1,
        norm=norm,
        read_noise=0.01,
        add_poisson_noise_to_data=False,
    )
    dataset = simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)
    dataset_list.append(dataset)

"""
__Model + Fit__
"""
model = af.Collection(
    cti=af.Model(
        ac.CTI1D,
        trap_list=[af.Model(ac.TrapInstantCapture)],
        ccd=af.Model(ac.CCDPhase, well_notch_depth=0.0, full_well_depth=200000.0),
    )
)

search = af.Nautilus(
    path_prefix=path.join("dataset_1d"),
    name="model_fit",
    unique_tag="integration",
    n_live=75,
)

analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=model, analysis=analysis)
    for analysis in analysis_list
]

factor_graph = af.FactorGraphModel(*analysis_factor_list)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("max log likelihood:", result_list[0].log_likelihood)
print(result_list[0].max_log_likelihood_instance.cti.trap_list[0].density)

fit = result_list[0].max_log_likelihood_fit
print("fit chi squared:", fit.chi_squared)

"""
__Aggregator round-trip__
"""
from pathlib import Path

from autocti import conf
from autocti import with_test_mode_segment

database_file = "dataset_1d_integration.sqlite"

if path.exists(database_file):
    os.remove(database_file)

agg = af.Aggregator.from_database(
    filename=database_file, completed_only=False, top_level_only=True
)
agg.add_directory(
    directory=str(with_test_mode_segment(Path(conf.instance.output_path)) / "dataset_1d")
)

dataset_agg = ac.agg.Dataset1DAgg(aggregator=agg)

total_datasets = 0
for dataset_sublist in dataset_agg.dataset_list_gen_from():
    total_datasets += len(dataset_sublist)

assert total_datasets >= len(norm_list), (
    f"aggregator returned {total_datasets} datasets, expected >= {len(norm_list)}"
)

fit_agg = ac.agg.FitDataset1DAgg(aggregator=agg)

total_fits = 0
for fit_sublist in fit_agg.max_log_likelihood_gen_from():
    total_fits += len(fit_sublist)

assert total_fits >= len(norm_list)

if path.exists(database_file):
    os.remove(database_file)

print("dataset_1d integration: OK")
