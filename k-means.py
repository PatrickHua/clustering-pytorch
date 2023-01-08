import torch
from torch import Tensor
from typing import Optional, Union, List

# @torch.no_grad()
def one_step_k_means(X: Tensor, C: Tensor) -> Tensor:
    """Performs one step kmeans.
    
    Args:
        X: 3d feature tensor of shape [batch_size, num_samples, feature_dim]
        C: current centroids [batch_size, num_centroids, feature_dim]
    
    Returns:
        C_new: Updated centroids [batch_size, num_centroids, feature_dim]
        W: assignment weight matrix [batch_size, num_samples, num_centroids]
    """

    distance = (X.unsqueeze(2) - C.unsqueeze(1)).pow(2).sum(dim=-1).argmin(dim=-1)  #.sqrt()

    _distance = distance[...,None]
    mask = (torch.arange(C.shape[1], dtype=X.dtype, device=X.device)[None, None] == _distance).type(X.dtype)
    
    nominator = (X.unsqueeze(-2) * mask.unsqueeze(-1)).sum(dim=1)
    denominator = mask.sum(dim=1)

    centroids = nominator / denominator.unsqueeze(-1)
    return centroids, distance

# @torch.no_grad()
def k_means(X: Tensor, iterations: int = 5, k: int = 4) -> Tensor:
    """Batched k-means algorithm"""
    batch_size, sample_size, feature_dim = X.shape

    C_ids = torch.randint(0, sample_size, size=(batch_size, k))

    C = torch.gather(X, dim=1, index=C_ids.unsqueeze(-1).repeat(1, 1, feature_dim))


    for t in range(iterations):
        C, W = one_step_k_means(X, C)

    return C, W


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from random import randint
    n_samples = 100
    centers = 4
    k = 4
    n_features = 2
    batch_size = 10
    it = 50


    xs = []
    for b in range(batch_size):
        X, _ = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=randint(0, 10000))
        xs.append(torch.tensor(X))
    
    X = torch.stack(xs)

    from time import time
    tic = time()

    centroids, assignment = k_means(X, it, k)

    import matplotlib.pyplot as plt

    for b in range(batch_size):
        plt.scatter(X[b,:,0], X[b,:,1], c=assignment[b])
        plt.show()



