# TODO: Add information about how the files are generated into this program.
# Importing the Framework and other important libraries
import sionna
import sionna.rt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from EnvironmentFramework import Environment, UAV, GroundUser
from sionna.rt import PlanarArray

import mitsuba
import drjit

print(tf.config.list_physical_devices("GPU"))
print(sionna.rt.__version__)
print(mitsuba.__version__)
print(drjit.__version__)

print("Done Printing")

env = sionna.rt.load_scene(sionna.rt.scene.simple_wedge)
print(f'Got the scene! {env.bandwidth}')