"""
Integration: cosmic-ray flagging on charge injection imaging
=============================================================

Simulates a charge injection image with CTI, read noise and a seeded cosmic-ray map, then runs the
flagging recipe of `autocti_workspace/scripts/imaging_ci/data_preparation/examples/cosmic_ray_flagging.py`
(correct the CTI, subtract the charge injection pattern, flag pixels above a signal-to-noise
threshold) and checks it against the cosmic rays that were injected:

 - every bright injected cosmic-ray pixel is flagged;
 - almost nothing that is not a cosmic ray is flagged;
 - the CTI-calibration mask built from the flags (`Mask2D.from_cosmic_ray_map_buffed`) covers every
   pixel the cosmic rays contaminate, including the trails CTI drags behind them, so fits never
   see cosmic-ray charge. "Contaminated" is measured directly, by clocking the image with and
   without the cosmic rays, not taken from the mask's own buffer settings.

The cosmic rays and the noise are seeded, so the result is deterministic. CTI is added and
removed by the same arctic, so the checks hold whatever arcticpy version is installed.
"""

import numpy as np

import autocti as ac

"""
__Dataset__
"""
shape_native = (100, 60)

parallel_overscan = ac.Region2D((90, 100, 5, 55))
serial_prescan = ac.Region2D((0, 100, 0, 5))
serial_overscan = ac.Region2D((0, 90, 55, 60))

layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=[(10, 40, 5, 55), (55, 85, 5, 55)],
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

"""
The Euclid-like trap model of the workspace's cosmic-ray flagging example. Trail length grows with
trap density and release timescale, so the mask's default buffers are sized for realistic
densities like these. Measured 2026-11-06 (arcticpy 2.6): at the much denser single-trap model
the other scripts use (density 10, release timescale 5), cosmic-ray trails stay above the read
noise for 11-17 rows, past the default 10-row parallel buffer, and 61 contaminated pixels go
unmasked.
"""
ccd = ac.CCDPhase(well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0)

cti = ac.CTI2D(
    parallel_trap_list=[
        ac.TrapInstantCapture(density=0.13, release_timescale=1.25),
        ac.TrapInstantCapture(density=0.25, release_timescale=4.4),
    ],
    parallel_ccd=ccd,
    serial_trap_list=[
        ac.TrapInstantCapture(density=0.0442, release_timescale=0.8),
        ac.TrapInstantCapture(density=0.1326, release_timescale=4.0),
        ac.TrapInstantCapture(density=3.9782, release_timescale=20.0),
    ],
    serial_ccd=ccd,
)

clocker = ac.Clocker2D(
    parallel_express=5,
    parallel_roe=ac.ROEChargeInjection(),
    serial_express=5,
    serial_roe=ac.ROE(),
)

read_noise = 4.0

cosmic_ray_map = ac.SimulatorCosmicRayMap.defaults(
    shape_native=shape_native, flux_scaling=1.0, pixel_scale=0.1, seed=1
).cosmic_ray_map_from(limit=ccd.full_well_depth)

assert np.sum(np.asarray(cosmic_ray_map) > 0.0) > 0, (
    "The seeded cosmic-ray simulation injected no cosmic rays: the test is vacuous"
)

dataset = ac.SimulatorImagingCI(
    read_noise=read_noise, pixel_scales=0.1, norm=1000.0, noise_seed=1
).via_layout_from(clocker=clocker, layout=layout, cti=cti, cosmic_ray_map=cosmic_ray_map)

"""
__Flag__

The recipe of the workspace example: correct the CTI, subtract the charge injection pattern so
only cosmic-ray charge (and noise) is left, then flag pixels above 4 sigma.
"""
cr_threshold = 4.0

data_corrected = clocker.remove_cti(data=dataset.data, cti=cti)
data_charge_subtracted = np.asarray(data_corrected.native) - np.asarray(
    dataset.pre_cti_data.native
)
flag = data_charge_subtracted > cr_threshold * np.asarray(dataset.noise_map.native)

injected = np.asarray(cosmic_ray_map.native) > 0.0
# "Bright" is fixed in electrons, independent of the flagging threshold, so a threshold
# regression cannot also lower the bar it is judged against.
bright = np.asarray(cosmic_ray_map.native) > 40.0 * read_noise

n_bright = int(np.sum(bright))
n_bright_flagged = int(np.sum(flag & bright))
n_false = int(np.sum(flag & ~injected))
n_clean = int(np.sum(~injected))

print(f"injected cosmic-ray pixels: {int(np.sum(injected))} ({n_bright} bright)")
print(f"bright cosmic-ray pixels flagged: {n_bright_flagged} / {n_bright}")
print(f"clean pixels flagged: {n_false} / {n_clean}")

assert n_bright > 0, "No bright cosmic-ray pixels were injected: the test is vacuous"
assert n_bright_flagged == n_bright, (
    f"{n_bright - n_bright_flagged} bright cosmic-ray pixels were not flagged"
)
assert n_false <= 0.01 * n_clean, (
    f"{n_false} clean pixels flagged, more than 1% of {n_clean}"
)

"""
__Mask__

The mask CTI calibration uses: each flagged pixel plus the parallel / serial / diagonal buffers
that cover the trail CTI drags behind a cosmic ray.
"""
settings = ac.SettingsMask2D(
    cosmic_ray_parallel_buffer=10,
    cosmic_ray_serial_buffer=10,
    cosmic_ray_diagonal_buffer=3,
)

mask = ac.Mask2D.from_cosmic_ray_map_buffed(
    cosmic_ray_map=ac.Array2D.no_mask(
        values=flag.astype("float"), pixel_scales=dataset.pixel_scales
    ),
    settings=settings,
)
mask = np.asarray(mask)

assert np.all(mask[flag]), "A flagged cosmic-ray pixel is not masked"

"""
Which pixels do the cosmic rays actually contaminate? Clock the noise-free charge injection
image with and without them: every pixel the cosmic-ray charge (or its CTI trail) changes by
more than the read noise must be masked.
"""
pre_cti_no_noise = np.asarray(dataset.pre_cti_data.native)

post_cti_without_cr = clocker.add_cti(data=dataset.pre_cti_data, cti=cti)
post_cti_with_cr = clocker.add_cti(
    data=ac.Array2D.no_mask(
        values=pre_cti_no_noise + np.asarray(cosmic_ray_map.native),
        pixel_scales=dataset.pixel_scales,
    ).native,
    cti=cti,
)
contaminated = (
    np.abs(np.asarray(post_cti_with_cr.native) - np.asarray(post_cti_without_cr.native))
    > read_noise
)

n_contaminated = int(np.sum(contaminated))
n_contaminated_unmasked = int(np.sum(contaminated & ~mask))
print(f"pixels contaminated by cosmic rays + trails: {n_contaminated}")
print(f"contaminated pixels left unmasked: {n_contaminated_unmasked}")

assert n_contaminated > n_bright, (
    "The cosmic rays leave no CTI trail beyond their own pixels: the trail check is vacuous"
)
assert n_contaminated_unmasked == 0, (
    f"{n_contaminated_unmasked} pixels contaminated by cosmic-ray charge or its CTI trail "
    f"are not masked"
)

print(f"masked pixels: {int(np.sum(mask))} / {mask.size}")
print("imaging_ci cosmic_ray_flagging: OK")
