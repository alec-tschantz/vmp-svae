import torch


def expected_log_pi(alpha: torch.Tensor):
    return torch.digamma(alpha) - torch.digamma(torch.sum(alpha, dim=-1, keepdim=True))


def standard_to_natural(alpha: torch.Tensor):
    return alpha - 1


def natural_to_standard(eta: torch.Tensor):
    return eta + 1
