import numpy as np
import chainer.functions as F
from chainer import Variable
from kernel import pdist, rbf


def test_pdist():
    x = Variable(np.array([
        [1], [2], [3],
    ], dtype='f'))
    y = Variable(np.array([
        [3], [2], [1],
    ], dtype='f'))
    res = pdist(x, y)
    res = F.sum(res)
    res.backward()
    np.testing.assert_allclose(x.grad, [[-6], [0], [6]])


def test_rbf():
    x = Variable(np.array([
        [1], [2],
    ], dtype='f'))

    res = rbf(x)
    np.testing.assert_allclose(res.data, [
        [1.0, 0.111111111],
        [0.111111111, 1.0],
    ])
