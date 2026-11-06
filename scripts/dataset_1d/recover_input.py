"""
Integration: Dataset1D recovers its input CTI model
====================================================

Simulates a noise-free 1D charge injection dataset from a known trap and CCD model, then checks
that the likelihood is a self-consistent function of the model that generated the data:

 - fitting the true model gives a chi-squared of (numerically) zero;
 - moving any parameter away from the truth lowers the log likelihood;
 - a scan of the log likelihood over trap density peaks exactly at the true density.

The data is simulated and fitted with the same arctic, so every assertion holds whatever
arcticpy version is installed. Absolute log-likelihood values are pinned separately, in
`likelihood/pins.py`.
"""

import numpy as np

import autofit as af
import autocti as ac

"""
__Dataset__
"""
layout = ac.Layout1D(
    shape_1d=(50,),
    region_list=[(5, 25)],
    prescan=ac.Region1D(region=(0, 5)),
    overscan=ac.Region1D(region=(45, 50)),
)

clocker = ac.Clocker1D(express=5, roe=ac.ROEChargeInjection())

density_true = 10.0
release_timescale_true = 5.0
well_fill_power_true = 0.5


def cti_from(
    density=density_true,
    release_timescale=release_timescale_true,
    well_fill_power=well_fill_power_true,
):
    return ac.CTI1D(
        trap_list=[
            ac.TrapInstantCapture(density=density, release_timescale=release_timescale)
        ],
        ccd=ac.CCDPhase(
            well_fill_power=well_fill_power,
            well_notch_depth=0.0,
            full_well_depth=200000.0,
        ),
    )


simulator = ac.SimulatorDataset1D(
    pixel_scales=0.1,
    norm=5000.0,
    read_noise=None,
    add_poisson_noise_to_data=False,
)
dataset = simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti_from())

analysis = ac.AnalysisDataset1D(dataset=dataset, clocker=clocker)


def instance_from(**kwargs):
    instance = af.ModelInstance()
    instance.cti = cti_from(**kwargs)
    return instance


def log_likelihood_from(**kwargs):
    return float(analysis.log_likelihood_function(instance=instance_from(**kwargs)))


"""
__Truth fits the data__
"""
fit_truth = analysis.fit_via_instance_from(instance=instance_from())

print(f"chi-squared at the truth: {fit_truth.chi_squared:.3e}")
assert fit_truth.chi_squared < 1e-6, (
    f"The true CTI model does not reproduce its own noise-free simulation: "
    f"chi-squared = {fit_truth.chi_squared}"
)

"""
__Every offset is worse__
"""
log_likelihood_truth = log_likelihood_from()

for name, kwargs in [
    ("density x1.2", dict(density=density_true * 1.2)),
    ("density x0.8", dict(density=density_true * 0.8)),
    ("release_timescale x1.2", dict(release_timescale=release_timescale_true * 1.2)),
    ("well_fill_power x1.2", dict(well_fill_power=well_fill_power_true * 1.2)),
]:
    log_likelihood_offset = log_likelihood_from(**kwargs)
    print(f"{name}: {log_likelihood_offset:.4f} vs truth {log_likelihood_truth:.4f}")
    assert log_likelihood_offset < log_likelihood_truth, (
        f"{name} fits at least as well as the truth "
        f"({log_likelihood_offset} >= {log_likelihood_truth})"
    )

"""
__Density scan recovers the input__
"""
density_grid = density_true * np.linspace(0.5, 1.5, 21)
log_likelihood_grid = [log_likelihood_from(density=density) for density in density_grid]
density_recovered = density_grid[int(np.argmax(log_likelihood_grid))]

print(f"recovered density: {density_recovered} (truth {density_true})")
assert np.isclose(density_recovered, density_true), (
    f"The likelihood scan peaks at density {density_recovered}, not the true {density_true}"
)

print("dataset_1d recover_input: OK")
