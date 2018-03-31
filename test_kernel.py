import numpy as np
import chainer.functions as F
from chainer import Variable
from kernel import pdist


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
