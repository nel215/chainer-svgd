import numpy as np
from svgd.model import GMM


def test_call():
    np.random.seed(215)
    mu = np.array([
        [-1, -1],
        [1, 1],
    ], dtype='f')
    sigma = np.array([
        [1.5, 0.5],
        [0.5, 1.5],
    ], dtype='f')
    x0 = np.random.randn(100, 2, 2).astype('f')
    model = GMM(mu, sigma, x0)
    loss = model()
    assert loss.shape == ()
