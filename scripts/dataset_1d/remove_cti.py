"""
Integration: Dataset1D CTI correction round trip
================================================

Adds CTI to a 1D charge injection line with arctic, then removes it with the same CTI model,
and checks the correction recovers the original data:

 - `remove_cti(add_cti(data))` matches `data` to within a small fraction of the injected signal;
 - the residual shrinks as the number of correction iterations grows.

The trails that CTI adds are large (tens of percent of the signal in the first EPER pixels), so
recovering the input to well below that is a real test of the correction. The same arctic adds
and removes the CTI, so the assertions hold whatever arcticpy version is installed.
"""

import numpy as np

import autocti as ac

"""
__Data__
"""
norm = 5000.0

layout = ac.Layout1D(
    shape_1d=(50,),
    region_list=[(5, 25)],
    prescan=ac.Region1D(region=(0, 5)),
    overscan=ac.Region1D(region=(45, 50)),
)

cti = ac.CTI1D(
    trap_list=[ac.TrapInstantCapture(density=10.0, release_timescale=5.0)],
    ccd=ac.CCDPhase(well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0),
)

clocker = ac.Clocker1D(express=5, roe=ac.ROEChargeInjection())

dataset = ac.SimulatorDataset1D(
    pixel_scales=0.1, norm=norm, read_noise=None, add_poisson_noise_to_data=False
).via_layout_from(clocker=clocker, layout=layout, cti=cti)

pre_cti_data = dataset.pre_cti_data
post_cti_data = clocker.add_cti(data=pre_cti_data, cti=cti)

trail_size = np.max(np.abs(np.asarray(post_cti_data) - np.asarray(pre_cti_data)))
print(f"largest CTI change to the data: {trail_size:.3f} e-")
assert trail_size > 0.01 * norm, "The CTI model barely changes the data: the test is vacuous"

"""
__Correct__
"""
residual_list = []

for iterations in [1, 5]:
    clocker = ac.Clocker1D(
        express=5, roe=ac.ROEChargeInjection(), iterations=iterations
    )
    corrected = clocker.remove_cti(data=post_cti_data, cti=cti)
    residual = np.max(np.abs(np.asarray(corrected) - np.asarray(pre_cti_data)))
    residual_list.append(residual)
    print(f"iterations={iterations}: max |corrected - input| = {residual:.4e} e-")

assert residual_list[1] < 1e-3 * trail_size, (
    f"5-iteration correction leaves a residual of {residual_list[1]} e-, "
    f"more than 0.1% of the {trail_size:.3f} e- CTI trail it should remove"
)
assert residual_list[1] < residual_list[0], (
    f"more iterations did not reduce the residual: {residual_list}"
)

print("dataset_1d remove_cti: OK")
