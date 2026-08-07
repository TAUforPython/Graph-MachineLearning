import torch
from healthcare_gnn.models.hyperbolic import distance, expmap0, logmap0, mobius_add, project

def test_operations_and_finite_gradients():
    c=torch.tensor(0.7); v=torch.tensor([[0.2,-0.1]],requires_grad=True)
    x=expmap0(v,c); recovered=logmap0(x,c)
    assert torch.allclose(v,recovered,atol=1e-5,rtol=1e-5)
    zero=torch.zeros_like(x); assert torch.allclose(mobius_add(x,zero,c),x,atol=1e-5)
    assert torch.allclose(distance(x,zero,c),distance(zero,x,c),atol=1e-5)
    distance(x,zero,c).sum().backward(); assert torch.isfinite(v.grad).all()
    assert torch.linalg.vector_norm(project(torch.tensor([[100.,0.]]),c))*torch.sqrt(c)<1
