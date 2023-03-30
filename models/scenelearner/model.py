import torch
import torch.nn as nn

from models.nn import *
from models.nn.box_registry import build_box_registry
from models.percept import *
from .executor import *
from utils import *

class UnknownArgument(Exception):
    def __init__():super()

class HierarchicalLearner(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        # [Unsupervised Part-Centric Representation]
        if config.perception == "slot_attention":
            if config.imsize == 128:self.scene_perception = SlotAttentionParser(config.num_slots,config.object_dim,config.slot_itrs)
            else:self.scene_perception = SlotAttentionParser64(config.num_slots,config.object_dim,config.slot_itrs)
        else:
            raise UnknownArgument

        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)

        # [Neuro Symbolic Executor]
        self.executor = HalProgramExecutor(config)

    def parse(self, program):return self.executor.parse(program)
    
    def forward(self, inputs):

        # [Parse the Input Scenes]
        part_centric_output = self.scene_perception(inputs["image"])

        # get the components
        full_recon = part_centric_output["full_recons"]
        recons     = part_centric_output["recons"]
        masks      = part_centric_output["masks"]
        loss       = part_centric_output["loss"]



        for question in inputs["question"]:
            for b in range(len(question["program"])):
                program = question["program"][b] # string program
                answer  = question["answer"][b]  # string answer

                scores   = part_centric_output["object_scores"][b,...,0] - EPS
                features = part_centric_output["object_features"][b]

                edge = 1e-4
                features = torch.cat([features,edge * torch.ones(features.shape)],-1)

                kwargs = {"features":features,
                                  "end":scores }

                q = self.executor.parse(program)
                        
                o = self.executor(q, **kwargs)
                print("Batch:{} P:{} A:{}".format(b,q,o))


        outputs = {**part_centric_output}
        return outputs

class SceneGraphLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # [Unsupervised Part-Centric Representation]
        if config.perception == "psgnet":
            self.scene_perception = 0
        else:
            raise UnknownArgument

        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)

        # [Neuro Symbolic Executor]
        self.executor = HalProgramExecutor(config)