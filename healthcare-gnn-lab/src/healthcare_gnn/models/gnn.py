"""Parameter-comparable Euclidean and tangent-aggregation graph models."""
from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
from .hyperbolic import expmap0, logmap0, project

class NodeGNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, kind: str = "gcn") -> None:
        super().__init__()
        factories={"gcn":GCNConv,"sage":SAGEConv,"gat":lambda a,b:GATConv(a,b,heads=1),"transformer":lambda a,b:TransformerConv(a,b,heads=1)}
        if kind not in factories: raise ValueError(f"Unknown GNN kind: {kind}")
        self.conv1=factories[kind](in_channels,hidden); self.conv2=factories[kind](hidden,hidden); self.head=nn.Linear(hidden,1)
    def forward(self,x:torch.Tensor,edge_index:torch.Tensor)->torch.Tensor:
        return self.head(self.conv2(torch.relu(self.conv1(x,edge_index)),edge_index)).squeeze(-1)

class HyperbolicGCN(nn.Module):
    """Poincaré lift with tangent-space GCN aggregation and tangent readout.

    This is a geometry-aware adaptation, not a fully intrinsic reproduction of a
    cited architecture. Curvature is positive via softplus and logged by callers.
    """
    def __init__(self,in_channels:int,hidden:int,initial_c:float=1.0)->None:
        super().__init__(); self.input=nn.Linear(in_channels,hidden); self.conv=GCNConv(hidden,hidden); self.head=nn.Linear(hidden,1)
        self.raw_c=nn.Parameter(torch.tensor(float(initial_c)).log())
    @property
    def curvature(self)->torch.Tensor: return torch.nn.functional.softplus(self.raw_c)+1e-5
    def forward(self,x:torch.Tensor,edge_index:torch.Tensor)->torch.Tensor:
        tangent=self.input(x); ball=expmap0(tangent,self.curvature)
        aggregated=self.conv(logmap0(ball,self.curvature),edge_index)
        ball_out=project(expmap0(torch.relu(aggregated),self.curvature),self.curvature)
        return self.head(logmap0(ball_out,self.curvature)).squeeze(-1)
