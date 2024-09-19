import torch
import numpy as np


def decay_lr(lr: float, global_step: int, decay_rate: int = 1, decay_steps: int = 1000):
    return lr * np.power(decay_rate, global_step / decay_steps)


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum("ti,tij->tj", features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))

    return torch.tensor(data[:, 0:2]).float(), torch.tensor(data[:, 2], dtype=torch.int64)


def rand_partial_isometry(input_dim, output_dim, stddev=1.0, seed=0):
    d = max(input_dim, output_dim)
    npr = np.random.RandomState(seed)
    return np.linalg.qr(npr.normal(loc=0, scale=stddev, size=(d, d)))[0][:input_dim, :output_dim]
