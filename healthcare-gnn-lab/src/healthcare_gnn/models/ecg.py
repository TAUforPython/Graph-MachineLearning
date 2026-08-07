"""Shared lead-wise temporal encoder for PTB-XL adaptations."""
from __future__ import annotations
import torch
from torch import nn
class SharedLeadEncoder(nn.Module):
    """Apply one 1-D CNN to every lead; input shape is [batch,12,time]."""
    def __init__(self,channels:int=16)->None:
        super().__init__(); self.network=nn.Sequential(nn.Conv1d(1,channels,7,stride=2,padding=3),nn.ReLU(),nn.Conv1d(channels,channels,5,stride=2,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,signal:torch.Tensor)->torch.Tensor:
        if signal.ndim!=3 or signal.shape[1]!=12: raise ValueError("signal must have shape [batch,12,time]")
        batch,leads,length=signal.shape
        return self.network(signal.reshape(batch*leads,1,length)).squeeze(-1).reshape(batch,leads,-1)
