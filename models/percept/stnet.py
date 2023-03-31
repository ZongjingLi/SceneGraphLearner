from types import SimpleNamespace

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.nn    import max_pool_x
from torch_geometric.utils import add_self_loops
from torch_scatter         import scatter_mean
from torch_sparse          import coalesce


from torch_sparse  import SparseTensor
from torch_scatter import scatter_max


from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch
from torch_scatter import scatter_mean,scatter_max

from primary import * 
from utils import *
