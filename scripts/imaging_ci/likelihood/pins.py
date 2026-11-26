"""
Likelihood Pins: ImagingCI
==========================

Pins the absolute value of `AnalysisImagingCI.log_likelihood_function` on a deterministic
simulated charge injection image, so a silent numerical change to the 2D CTI likelihood
(arctic clocking -> FPR / EPER extraction -> chi-squared) fails CI instead of passing
unnoticed.

Three clocking directions are pinned: parallel only, serial only, and parallel + serial. Each
is evaluated at:

- **truth**: the instance used to simulate the data. With no read noise the chi-squared is
  zero, so this value pins the noise normalization.
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

The layout of `scripts/imaging_ci/model_fit.py`, with no noise added. The noise map takes the
simulator's constant fallback value, so chi-squared is well defined.
"""
shape_native = (30, 30)

parallel_overscan = ac.Region2D((25, 30, 1, 29))
serial_prescan = ac.Region2D((0, 30, 0, 1))
serial_overscan = ac.Region2D((0, 25, 29, 30))

layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=[(0, 25, serial_prescan[3], serial_overscan[2])],
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

ccd = ac.CCDPhase(well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0)

clocker_dict = {
    "parallel": ac.Clocker2D(parallel_express=5, parallel_roe=ac.ROEChargeInjection()),
    "serial": ac.Clocker2D(serial_express=5, serial_roe=ac.ROE()),
    "parallel_serial": ac.Clocker2D(
        parallel_express=5,
        parallel_roe=ac.ROEChargeInjection(),
        serial_express=5,
        serial_roe=ac.ROE(),
    ),
}


def cti_from(direction, scale_density=1.0, scale_timescale=1.0):
    parallel_trap_list = [
        ac.TrapInstantCapture(
            density=10.0 * scale_density, release_timescale=5.0 * scale_timescale
        )
    ]
    serial_trap_list = [
        ac.TrapInstantCapture(
            density=5.0 * scale_density, release_timescale=2.0 * scale_timescale
        )
    ]

    if direction == "parallel":
        return ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=ccd)
    if direction == "serial":
        return ac.CTI2D(serial_trap_list=serial_trap_list, serial_ccd=ccd)
    return ac.CTI2D(
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=ccd,
    )


def log_likelihood_from(direction, scale_density, scale_timescale):
    """
    `log_likelihood_function` with data simulated from the truth model and evaluated at the
    (possibly offset) model.
    """
    clocker = clocker_dict[direction]

    simulator = ac.SimulatorImagingCI(read_noise=None, pixel_scales=0.1, norm=100.0)
    dataset = simulator.via_layout_from(
        clocker=clocker, layout=layout, cti=cti_from(direction=direction)
    )

    instance = af.ModelInstance()
    instance.cti = cti_from(
        direction=direction,
        scale_density=scale_density,
        scale_timescale=scale_timescale,
    )

    analysis = ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)

    return float(analysis.log_likelihood_function(instance=instance))


"""
__Pins__

Each row is (direction, point, model kwargs, pinned log likelihood).
"""
# pinned 2026-11-04: autocti main@2a91ff5d, arcticpy 2.6 (autocti_workspace_test#24).
# verified 2026-11-26: unchanged under arcticpy 2.7 (jkeger/arctic@6f51535, PyAutoHeart#372).
offset = dict(scale_density=1.2, scale_timescale=0.8)
truth = dict(scale_density=1.0, scale_timescale=1.0)

pins = [
    ("parallel", "truth", truth, 1245.2819038104358),
    ("parallel", "offset", offset, -1804.1714185088567),
    ("serial", "truth", truth, 1245.2819038104358),
    ("serial", "offset", offset, 942.9536414447045),
    ("parallel_serial", "truth", truth, 1245.2819038104358),
    ("parallel_serial", "offset", offset, -2110.2378525975064),
]

for direction, point, kwargs, expected in pins:
    value = log_likelihood_from(direction=direction, **kwargs)
    print(f"imaging_ci {direction}, {point}: {value!r}")
    np.testing.assert_allclose(
        value,
        expected,
        rtol=1e-8,
        err_msg=f"imaging_ci likelihood pin moved: {direction}, {point}",
    )

print("imaging_ci likelihood pins: OK")
