"""
Integration: ImagingCI recovers its input CTI model
====================================================

Simulates noise-free charge injection images from a known CTI model, for parallel CTI, serial
CTI and both together, and checks the likelihood is self-consistent with the model that
generated the data:

 - fitting the true model gives a chi-squared of (numerically) zero;
 - moving the trap density or release timescale away from the truth lowers the log likelihood;
 - a scan of the log likelihood over trap density peaks exactly at the true density.

The data is simulated and fitted with the same arctic, so every assertion holds whatever
arcticpy version is installed. Absolute log-likelihood values are pinned separately, in
`likelihood/pins.py`, whose layout this script shares.
"""

import numpy as np

import autofit as af
import autocti as ac

"""
__Dataset__
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

parallel_density_true = 10.0
serial_density_true = 5.0


def cti_from(direction, scale_density=1.0, scale_timescale=1.0):
    parallel_trap_list = [
        ac.TrapInstantCapture(
            density=parallel_density_true * scale_density,
            release_timescale=5.0 * scale_timescale,
        )
    ]
    serial_trap_list = [
        ac.TrapInstantCapture(
            density=serial_density_true * scale_density,
            release_timescale=2.0 * scale_timescale,
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


for direction, clocker in clocker_dict.items():
    simulator = ac.SimulatorImagingCI(read_noise=None, pixel_scales=0.1, norm=5000.0)
    dataset = simulator.via_layout_from(
        clocker=clocker, layout=layout, cti=cti_from(direction=direction)
    )

    analysis = ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)

    def instance_from(**kwargs):
        instance = af.ModelInstance()
        instance.cti = cti_from(direction=direction, **kwargs)
        return instance

    def log_likelihood_from(**kwargs):
        return float(analysis.log_likelihood_function(instance=instance_from(**kwargs)))

    """
    __Truth fits the data__
    """
    fit_truth = analysis.fit_via_instance_from(instance=instance_from())

    print(f"{direction}: chi-squared at the truth {fit_truth.chi_squared:.3e}")
    assert fit_truth.chi_squared < 1e-6, (
        f"{direction}: the true CTI model does not reproduce its own noise-free "
        f"simulation: chi-squared = {fit_truth.chi_squared}"
    )

    """
    __Every offset is worse__
    """
    log_likelihood_truth = log_likelihood_from()

    for name, kwargs in [
        ("density x1.2", dict(scale_density=1.2)),
        ("density x0.8", dict(scale_density=0.8)),
        ("release_timescale x1.2", dict(scale_timescale=1.2)),
    ]:
        log_likelihood_offset = log_likelihood_from(**kwargs)
        assert log_likelihood_offset < log_likelihood_truth, (
            f"{direction}: {name} fits at least as well as the truth "
            f"({log_likelihood_offset} >= {log_likelihood_truth})"
        )

    """
    __Density scan recovers the input__
    """
    scale_grid = np.linspace(0.5, 1.5, 11)
    log_likelihood_grid = [
        log_likelihood_from(scale_density=scale) for scale in scale_grid
    ]
    scale_recovered = scale_grid[int(np.argmax(log_likelihood_grid))]

    print(f"{direction}: recovered density scale {scale_recovered} (truth 1.0)")
    assert np.isclose(scale_recovered, 1.0), (
        f"{direction}: the likelihood scan peaks at density x{scale_recovered}, "
        f"not at the truth"
    )

print("imaging_ci recover_input: OK")
