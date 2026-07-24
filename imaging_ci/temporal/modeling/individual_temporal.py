"""
Temporal: Individual Temporal
=============================

In the script `individual_fits.py` we fitted multiple charge injection datasets one-by-one.

These datasets were simulated to represent different times in a space mission, such that the density of traps
linearly increases across the datasets.

In this script we load the results of these model fits and estimate how the density evolves linearly with time.
"""
import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af
import autocti as ac

from autofit.aggregator.aggregator import Aggregator

agg = Aggregator(directory="output")

"""
Using the database and aggregator, we can now compute lists of the estimate density of traps for every model fit,
including errors, alongside the number of months into the space mission the fitted dataset corresponds too.
"""
samples_list = list(agg.values("samples"))

density_list = [
    samples.median_pdf().cti.parallel_trap_list[0].density for samples in samples_list
]

ue3_instances = [samp.errors_at_upper_sigma(sigma=3.0) for samp in samples_list]
density_ue3_list = [
    instance.cti.parallel_trap_list[0].density for instance in ue3_instances
]

le3_instances = [samp.errors_at_lower_sigma(sigma=3.0) for samp in samples_list]
density_le3_list = [
    instance.cti.parallel_trap_list[0].density for instance in le3_instances
]

info_list = list(agg.values("info"))
month_list = [info["months"] for info in info_list]

plt.errorbar(
    x=month_list,
    y=density_list,
    marker=".",
    linestyle="",
    yerr=[density_le3_list, density_ue3_list],
)
plt.show()
plt.close()
