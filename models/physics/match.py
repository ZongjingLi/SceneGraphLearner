import itertools
import numpy as np

#TODO: from physics.loss import l2

class Matcher(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.penalty = {
            "type": config.type_penalty,
            "distance":config.distance_penalty,
            "color": config.color_penalty,
        }

        self.distance_threshold = config.distance_threshold
        self.base_penalty = config.base_penalty

    # TODO: config: type_penalty, distance_penalty, distance_threshold, base_penalty

    def _criterion(self, obj_1, obj_2):
        """
        Compute a location_loss between obj_1 and obj_2. Factors considered:
            - object 'type'
            - 3d location
        """