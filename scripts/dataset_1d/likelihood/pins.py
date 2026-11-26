"""
Likelihood Pins: Dataset1D
==========================

Pins the absolute value of `AnalysisDataset1D.log_likelihood_function` on a deterministic
simulated dataset, so a silent numerical change to the 1D CTI likelihood (arctic clocking ->
FPR / EPER extraction -> chi-squared) fails CI instead of passing unnoticed.

Two models are pinned, one with one trap species and one with two. Each is evaluated at:

- **truth**: the instance used to simulate the data. With no read noise and no Poisson noise
  the chi-squared is zero, so this value pins the noise normalization.
- **offset**: trap densities x1.2 and release timescales x0.8. The clocked data no longer
  match, so this value pins the arctic clocking and the chi-squared path.

This is the NumPy path only. Clocking runs through arcticpy's C++ code, so there is no JAX leg.

__Re-pinning__

A pin changes only with a dated comment that names the PR and the cause (see `AGENTS.md`).
The arcticpy version is part of each pin's provenance, because a new arcticpy release changes
the clocked values.
"""

import numpy as np

import autofit as af
import autocti as ac

"""
__Dataset__

Two injection levels, matching `scripts/dataset_1d/model_fit.py`, with no noise added. The
noise map takes the simulator's constant fallback value, so chi-squared is well defined.
"""
shape_native = (50,)

layout = ac.Layout1D(
    shape_1d=shape_native,
    region_list=[(5, 25)],
    prescan=ac.Region1D(region=(0, 5)),
    overscan=ac.Region1D(region=(45, 50)),
)

clocker = ac.Clocker1D(express=5, roe=ac.ROEChargeInjection())

ccd = ac.CCDPhase(well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0)

norm_list = [100.0, 5000.0]


def trap_list_from(scale_density=1.0, scale_timescale=1.0, species=1):
    trap_list = [
        ac.TrapInstantCapture(
            density=10.0 * scale_density, release_timescale=5.0 * scale_timescale
        )
    ]
    if species == 2:
        trap_list.append(
            ac.TrapInstantCapture(
                density=3.0 * scale_density, release_timescale=20.0 * scale_timescale
            )
        )
    return trap_list


def log_likelihood_from(species, scale_density, scale_timescale):
    """
    Sum of `log_likelihood_function` over the two injection levels, with data simulated
    from the truth model and evaluated at the (possibly offset) model.
    """
    cti_truth = ac.CTI1D(trap_list=trap_list_from(species=species), ccd=ccd)
    cti_eval = ac.CTI1D(
        trap_list=trap_list_from(
            scale_density=scale_density,
            scale_timescale=scale_timescale,
            species=species,
        ),
        ccd=ccd,
    )

    instance = af.ModelInstance()
    instance.cti = cti_eval

    total = 0.0

    for norm in norm_list:
        simulator = ac.SimulatorDataset1D(
            pixel_scales=0.1,
            norm=norm,
            read_noise=None,
            add_poisson_noise_to_data=False,
        )
        dataset = simulator.via_layout_from(
            clocker=clocker, layout=layout, cti=cti_truth
        )
        analysis = ac.AnalysisDataset1D(dataset=dataset, clocker=clocker)
        total += float(analysis.log_likelihood_function(instance=instance))

    return total


"""
__Pins__

Each row is (label, point, model kwargs, pinned log likelihood).
"""
# pinned 2026-11-04: autocti main@2a91ff5d, arcticpy 2.6 (autocti_workspace_test#24).
# verified 2026-11-26: unchanged under arcticpy 2.7 (jkeger/arctic@6f51535, PyAutoHeart#372).
offset = dict(scale_density=1.2, scale_timescale=0.8)
truth = dict(scale_density=1.0, scale_timescale=1.0)

pins = [
    ("x1", "truth", dict(species=1, **truth), 138.3646559789373),
    ("x1", "offset", dict(species=1, **offset), -16244.662767009368),
    ("x2", "truth", dict(species=2, **truth), 138.3646559789373),
    ("x2", "offset", dict(species=2, **offset), -25735.981752573927),
]

for label, point, kwargs, expected in pins:
    value = log_likelihood_from(**kwargs)
    print(f"dataset_1d {label} species, {point}: {value!r}")
    np.testing.assert_allclose(
        value,
        expected,
        rtol=1e-8,
        err_msg=f"dataset_1d likelihood pin moved: {label} species, {point}",
    )

print("dataset_1d likelihood pins: OK")
