"""
Integration: ImagingCI CTI correction round trip
================================================

Adds parallel and serial CTI to a charge injection image with arctic, then removes it with the
same CTI model, and checks the correction recovers the original image:

 - `remove_cti(add_cti(image))` matches `image` to well below the size of the CTI trails;
 - the residual shrinks as the number of correction iterations grows.

This is the CTI correction a user applies to science images after calibrating the CTI model.
The same arctic adds and removes the CTI, so the assertions hold whatever arcticpy version is
installed.
"""

import numpy as np

import autocti as ac

"""
__Data__
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

parallel_trap_list = [ac.TrapInstantCapture(density=10.0, release_timescale=5.0)]
serial_trap_list = [ac.TrapInstantCapture(density=5.0, release_timescale=2.0)]

cti_dict = {
    "parallel": ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=ccd),
    "serial": ac.CTI2D(serial_trap_list=serial_trap_list, serial_ccd=ccd),
    "parallel_serial": ac.CTI2D(
        parallel_trap_list=parallel_trap_list,
        parallel_ccd=ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=ccd,
    ),
}


def clocker_from(iterations):
    return ac.Clocker2D(
        iterations=iterations,
        parallel_express=5,
        parallel_roe=ac.ROEChargeInjection(),
        serial_express=5,
        serial_roe=ac.ROE(),
    )


pre_cti_data = (
    ac.SimulatorImagingCI(read_noise=None, pixel_scales=0.1, norm=5000.0)
    .via_layout_from(clocker=clocker_from(iterations=5), layout=layout, cti=None)
    .pre_cti_data
)

for direction, cti in cti_dict.items():
    post_cti_data = clocker_from(iterations=5).add_cti(data=pre_cti_data, cti=cti)

    trail_size = np.max(np.abs(np.asarray(post_cti_data) - np.asarray(pre_cti_data)))
    print(f"{direction}: largest CTI change to the image {trail_size:.3f} e-")
    assert trail_size > 1.0, (
        f"{direction}: the CTI model barely changes the image, so the test is vacuous"
    )

    """
    __Correct__
    """
    residual_list = []

    for iterations in [1, 5]:
        corrected = clocker_from(iterations=iterations).remove_cti(
            data=post_cti_data, cti=cti
        )
        residual_list.append(np.abs(np.asarray(corrected) - np.asarray(pre_cti_data)))
        print(
            f"{direction}: iterations={iterations} max |corrected - input| = "
            f"{residual_list[-1].max():.4e} e-, mean {residual_list[-1].mean():.4e} e-"
        )

    assert residual_list[1].max() < residual_list[0].max(), (
        f"{direction}: more iterations did not reduce the residual"
    )

    if direction != "parallel_serial":
        assert residual_list[1].max() < 1e-3 * trail_size, (
            f"{direction}: 5-iteration correction leaves a residual of "
            f"{residual_list[1].max()} e-, more than 0.1% of the {trail_size:.3f} e- "
            f"CTI trail it should remove"
        )
    else:
        """
        Parallel and serial CTI together do not correct to the same precision. Measured
        2026-11-06 (arcticpy 2.6): the residual plateaus at about 0.2 e- however many
        iterations are used (0.34 e- at 5, 0.23 e- at 10, 0.20 e- at 20), and always at a
        single pixel, the first row's serial-overscan corner (row 0, column 29). Each
        direction on its own converges to about 1e-6 e-. The bounds below therefore check
        that the image as a whole is corrected (mean residual) and that the corner stays
        small (max residual under 1% of the trail), rather than asserting the per-direction
        precision.
        """
        assert residual_list[1].mean() < 1e-3 * trail_size, (
            f"{direction}: mean residual {residual_list[1].mean()} e- is more than 0.1% "
            f"of the {trail_size:.3f} e- CTI trail"
        )
        assert residual_list[1].max() < 1e-2 * trail_size, (
            f"{direction}: max residual {residual_list[1].max()} e- is more than 1% "
            f"of the {trail_size:.3f} e- CTI trail"
        )

print("imaging_ci remove_cti: OK")
