# This is the Test script for the test case of trying to have
# a quad coptor fly through a narrow vertical slit in a wall
# the solver is the NMPC (non-linear model predictive controller) with
# the optimizer for hig dimensional problems as we also include difficult
# to evaluate cost functions (We want this to handle arbitrary cost function
# setups), in that vain we need optimizers that see the cost function as a
# black box and do not require gradients or hessians especially considering
# we want to consider discontinuous cost functions.

import matplotlib.pyplot as plt
import numpy as np

from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from dependencies.gncpy.src.gncpy.dynamics.aircraft import SimpleMultirotor

# configuration setup (Note we are in NED frame - this makes dynamics frame conversion easier for me)
dim = [0, 30, -7.5, 7.5, 0, -15]

GRAVITY = np.array([0,0,9.81])  # acceleration vector in NED frame

# vehicle info (fairly fast quad)
WEIGHT = 0.8  # kg


# State is 12 Dof (pos, vel, euler angles, angular rates) maybe the mixed representaiton idk
x0 = np.array([2.0, -2.0, -5, 0,0,0, 0,0,0, 0,0,0])  # initial state
u0 = np.array([0,0,0,0])  # initial input This actually need to be the hover

# make the configuration space
config_space = ConfigurationSpace3D(dim)