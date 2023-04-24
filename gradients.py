import torch
import torch.nn as nn

from config import *
from models import *

model = SceneLearner(config)
model.scene_perception = torch.load("/content/checkpoints/PTRObjects_toy_slot_attention.ckpt", map_location=args.device)
