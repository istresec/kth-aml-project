import numpy as np
from scipy.special import logsumexp


def compute_loglikelihood(model, loss_function, x, n_samples=5000):
    lls_block = None  # Will be a block/matrix of shape (n_samples, n_x)

    for s in range(n_samples):
        _, losses = loss_function(model, x)
        elbos = -np.asarray(losses)

        if lls_block is None:
            lls_block = np.zeros((n_samples, losses.shape[0]))
        lls_block[s] = elbos

    lls = logsumexp(lls_block, axis=0, return_sign=False) / n_samples
    if np.any(np.isnan(lls)):
        raise Exception(f"Inconsistent state, got a NaN loglikelihood! lls_block was: {lls_block}")
    return np.mean(lls)
