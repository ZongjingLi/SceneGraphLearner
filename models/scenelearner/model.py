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
            self.scene_perception = SceneGraphNet(config)
        if config.perception == "local_psgnet":
            self.scene_perception = ControlSceneGraphNet(config)
        if config.perception == "valkyr":
            self.scene_perception = ContrastNet(config) #EisenNet(config) #ConstructNet(config)
        if config.perception == "slot_attention":
            self.scene_perception = SlotAttentionParser(config.object_num, config.object_dim,5)
            self.part_perception = SlotAttention(config.part_num, config.object_dim,5)

        # PointNet Perception Module
        if config.perception in ["point_net","dgcnn"]:
            self.scene_perception = HierarchicalVNN(config)
        if config.perception in ["csqnet"]:
            self.scene_perception = CSQNet(config)

        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)

        # [Neuro Symbolic Executor]
        self.executor = SceneProgramExecutor(config)
        self.rep = config.concept_type

    def _check_nan_gradient(self):

        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).sum() > 0:
                    return True 
                    break
        return False 
        
    def parse(self, program):return self.executor.parse(program)
    
    def forward(self, inputs, query = None):

        # [Parse the Input Scenes]
        scene_tree_output = self.scene_perception(inputs)

        # get the components
