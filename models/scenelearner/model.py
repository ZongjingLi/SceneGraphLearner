import torch
import torch.nn as nn

from models.nn import *
from models.nn.box_registry import build_box_registry
from models.percept import *
from .executor import *
from utils import *

class UnknownArgument(Exception):
    def __init__():super()

class SceneLearner(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        # [Unsupervised Part-Centric Representation]
        if config.perception == "psgnet":
            self.scene_perception = SceneTreeNet(config)
        if config.perception == "slot_attention":
            self.scene_perception = SlotAttentionParser(config.object_num, config.object_dim,5)
            self.part_perception = SlotAttention(config.part_num, config.object_dim,5)

        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)

        # [Neuro Symbolic Executor]
        self.executor = SceneProgramExecutor(config)

    def parse(self, program):return self.executor.parse(program)
    
    def forward(self, inputs):

        # [Parse the Input Scenes]
        part_centric_output = self.scene_perception(inputs["image"])

        # get the components
