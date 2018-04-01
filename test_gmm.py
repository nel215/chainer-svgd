import numpy as np
from gmm import GMM


def _sample(n=50):
    mean = [[-1, -1], [1, 1]]
    var = [
        [
            [1, 0.0],
            [0, 0.8],
        ],
        [
            [1, 0],
            [0, 1.5],
        ],
    ]

    k = 0 if np.random.random() < 0.4 else 1
    res = []
    res = np.random.multivariate_normal(mean[k], var[k], n)

    return np.array(res, dtype='f')


def test_logp():
    np.random.seed(215)
    x = _sample()
    model = GMM()
    logp = model.logp(x)
    assert len(logp) == model.n_particle
