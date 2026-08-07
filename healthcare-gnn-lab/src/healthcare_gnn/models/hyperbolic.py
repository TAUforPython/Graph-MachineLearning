"""Numerically guarded Poincaré-ball operations using curvature ``-c``."""
from __future__ import annotations
import torch


def project(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    c=c.clamp_min(eps); maximum=(1.0-eps)/torch.sqrt(c)
    norm=x.norm(dim=-1,keepdim=True).clamp_min(eps)
    return x * torch.minimum(torch.ones_like(norm), maximum/norm)


def expmap0(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    c=c.clamp_min(eps); norm=v.norm(dim=-1,keepdim=True).clamp_min(eps); root=torch.sqrt(c)
    return project(torch.tanh(root*norm)*v/(root*norm),c)


def logmap0(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    c=c.clamp_min(eps); x=project(x,c,eps); norm=x.norm(dim=-1,keepdim=True).clamp_min(eps); root=torch.sqrt(c)
    scaled=(root*norm).clamp(max=1.0-eps)
    return torch.atanh(scaled)*x/(root*norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    c=c.clamp_min(eps); x2=(x*x).sum(-1,keepdim=True); y2=(y*y).sum(-1,keepdim=True); xy=(x*y).sum(-1,keepdim=True)
    denominator=(1+2*c*xy+c*c*x2*y2).clamp_min(eps)
    return project(((1+2*c*xy+c*y2)*x+(1-c*x2)*y)/denominator,c,eps)


def distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    delta=mobius_add(-x,y,c,eps); root=torch.sqrt(c.clamp_min(eps)); z=(root*delta.norm(dim=-1)).clamp(max=1.0-eps)
    return 2*torch.atanh(z)/root
