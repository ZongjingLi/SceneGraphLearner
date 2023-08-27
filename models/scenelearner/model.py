from turtle import shape
import torch
import torch.nn as nn

from models.nn import *
from models.nn.box_registry import build_box_registry
from models.percept import *
from .hierarchy_net import *
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
            self.scene_perception = None
            self.part_perception = ValkyrNet(config)
            
            #self.scene_perception = ValkyrNet(config) #EisenNet(config) #ConstructNet(config)
        if config.perception == "slot_attention":
            self.scene_perception = SlotAttentionParser(config.object_num, config.object_dim,5)
            self.part_perception = SlotAttention(config.part_num, config.object_dim,5)

        # PointNet Perception Module
        if config.perception in ["point_net","dgcnn"]:
            self.scene_perception = None
            self.part_perception = HierarchicalVNN(config)
        if config.perception in ["csqnet"]:
            self.scene_perception = None
            self.part_perception = CSQNet(config)

        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)
        # [Neuro Symbolic Executor]
        self.executor = SceneProgramExecutor(config)
        self.rep = config.concept_type

        # [Hierarchy Structure Network]
        self.scene_builder = nn.ModuleList([HierarchyBuilder(config, slot_num) \
            for slot_num in config.hierarchy_construct])
        self.is_box = False
        if self.is_box:
            self.hierarhcy_maps = nn.ModuleList([FCBlock(128,2,config.concept_dim,config.concept_dim)\
            for _ in range(len(config.hierarchy_construct)) ])
        else:
            self.hierarhcy_maps = nn.ModuleList([FCBlock(128,2,config.concept_dim,config.concept_dim)\
            for _ in range(len(config.hierarchy_construct)) ])
        self.feature2concept = nn.Linear(131,config.concept_dim)
        self.effective_level = "all"
        self.box_dim = config.concept_dim
    
    
    def build_scene(self,input_features):
        """
        features: BxNxD
        scores:   BxNx1 
        """
        B,N,D = input_features.shape
        scores = []
        features  = []
        connections = []
        box_dim = self.box_dim

        input_scores = torch.ones([B,N,1])
        for i,builder in enumerate(self.scene_builder):
            #input_features [B,N,D]
            if self.is_box:
                masks = builder(input_features[:,:,:box_dim], input_scores, self.executor) # [B,M,N]
            else:masks = builder(input_features, input_scores, self.executor) # [B,M,N]
            # [Build Scene Hierarchy]

            score = torch.max(masks, dim = -1).values.clamp(EPS,1-EPS) # hierarchy scores # [B,M]
            #print(score,masks)
            input_scores = score.unsqueeze(-1)
 
            feature = torch.einsum("bmn,bnd->bmd",masks,input_features) # hierarchy features # [B,M,D]
            if self.is_box:
                feature = self.hierarhcy_maps[i](feature[:,:,:box_dim])
                feature = torch.cat([feature, torch.ones([1,feature.shape[1],box_dim]) * EPS],dim = -1)
            else:
                feature = self.hierarhcy_maps[i](feature)
            # [Build Scores, Features, and Connections]
            scores.append(score) # [B,M]
            features.append(feature) # [B,M,D]
            connections.append(masks) # [B,M,N]

            input_features = feature

        scene_struct = {"scores":scores,"features":features,"connections":connections}
        return scene_struct

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
