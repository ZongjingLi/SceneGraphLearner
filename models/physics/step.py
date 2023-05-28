from copy import deepcopy
import numpy as np
import math

from .sim.step_sim import step, reverse_step

class Stepper(object):
    def __init__(self, config):
        perturbation_config = 0
        self.to_perturb = perturbation_config

        self.use_magic = True

        if self.use_magic:
            self.disappear_probability = 0.0
            self.disappear_penalty = 1.0
            
            self.stop_probability = 0.0
            self.stop_penalty = 1.0

            self.accelerate_probability = 0.0
            self.accelerate_penalty = 1.0

            self.accelerate_lambda = 0.5
    