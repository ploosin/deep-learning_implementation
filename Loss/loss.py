import torch

def gradient_penalty(y, x, device='cpu'):
    weight = torch.ones(y.size()).to(device)
    
    dydx = torch.autograd.grad(y, x, weight, True, True, True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2) 