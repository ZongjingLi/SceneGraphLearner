import torch
import torch.nn as nn
import torchvision

class BaseAttributes(object):
    
    @staticmethod
    def cum_sum(sequence):
        r,s = [0], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __init__(self, attributes_config):
        super(BaseAttributes, self).__init__()
        s