"""
Temporal: Individual Fits
=========================

In this script, we will fit multiple charge injection imaging to calibrate CTI, where:

 - The CTI model consists of one parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 - The `ImagingCI` is simulated with uniform charge injection lines and no cosmic rays.

The multiple datasets are representative of CTI calibration data taken over the course of a space mission. Therefore,
the model-fitting aims to determine the increase in the density of traps with time.

This script fits each dataset one-by-one and uses the results post-analysis to determine the density evolution
parameters. Other scripts perform more detailed analyses that use more advanced statistical methods to provide a
better estimate.

The charge injection data is a small cut-out of 30 x 30 pixels, to make CTI calibration run fast for the overview
script that this data is used for.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autocti as ac

"""
__Dataset__

The paths pointing to the dataset we will use for cti modeling.
"""
dataset_label = "temporal"
dataset_type = "parallel_x1"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/overview/uniform'
"""
dataset_path = path.join(dataset_label, "dataset", dataset_type)

"""
__Database__

First, note how the results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `database.sqlite` file, which we can load using the `Aggregator`.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator(directory=path.join("output", "temporal"))

max_lh_instance_list = [samps.max_log_likelihood() for samps in agg.values("samples")]

interpolator = af.LinearInterpolator(instances=max_lh_instance_list)
instance = interpolator[interpolator.time == 1.5]

print(instance.cti.parallel_trap_list[0].density)

"""
__Serialization__

Make sure that interpolator + model can be serialized to a .json file.
"""
json_file = path.join(dataset_path, "interpolator.json")

interpolator.output_to_json(file_path=json_file)

interpolator = af.LinearInterpolator.from_json(file_path=json_file)