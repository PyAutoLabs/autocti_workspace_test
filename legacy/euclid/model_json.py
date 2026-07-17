import autocti as ac

parallel_trap_list = [
    ac.TrapInstantCapture(density=0.1656, release_timescale=1.25),
    ac.TrapInstantCapture(density=0.3185, release_timescale=4.4),
]
parallel_ccd = ac.CCDPhase(
    well_notch_depth=1e-9, well_fill_power=0.58, full_well_depth=200000
)

serial_trap_list = [
    ac.TrapInstantCapture(density=0.0442, release_timescale=0.8),
    ac.TrapInstantCapture(density=0.1326, release_timescale=2.5),
    ac.TrapInstantCapture(density=3.9782, release_timescale=20.0),
]
serial_ccd = ac.CCDPhase(
    well_notch_depth=1e-9, well_fill_power=0.58, full_well_depth=200000
)

cti = ac.CTI2D(
    parallel_trap_list=parallel_trap_list,
    parallel_ccd=parallel_ccd,
    serial_trap_list=serial_trap_list,
    serial_ccd=serial_ccd,
)

cti.output_to_json(file_path="cti_model.json")

# ac.CTI2D.from_json(file_path="cti_model.json")
