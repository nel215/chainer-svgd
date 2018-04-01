import numpy as np
import chainer.functions as F


def pdist(x, y):
    n = x.shape[0]
    xx = F.broadcast_to(F.sum(F.square(x), axis=1), (n, n)).T
    xy = F.matmul(y, x, transb=True)
    yy = F.broadcast_to(F.sum(F.square(y), axis=1), (n, n))
    return xx - 2 * xy + yy


def rbf(x):
    dist = pdist(x, x.data)
    m = np.median(dist.data)  # cut graph
    w = m / np.log(x.shape[0])

    res = F.exp(-dist/w/2)

    return res
