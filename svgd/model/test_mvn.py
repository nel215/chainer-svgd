import numpy as np
from svgd.model import MVN


def test_logp():
    np.random.seed(215)
    mu = np.array([1.0, 2.0])
    sigma = np.array([1.0, 1.0])
    x0 = np.random.randn(100, 2)
    model = MVN(mu, sigma, x0)
    logp = model.logp()
    assert logp.shape == (100, 100)


def test_call():
    np.random.seed(215)
    mu = np.array([1.0, 2.0])
    sigma = np.array([1.0, 1.0])
    x0 = np.random.randn(100, 2)
    model = MVN(mu, sigma, x0)
    loss = model()
    assert loss.shape == ()
