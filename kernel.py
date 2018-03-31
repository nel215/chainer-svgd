import chainer.functions as F


def pdist(x, y):
    n = x.shape[0]
    xx = F.broadcast_to(F.square(x), (n, n)).T
    xy = F.matmul(y, x, transb=True)
    yy = F.broadcast_to(F.square(y), (n, n)).T
    return xx - 2 * xy + yy


def rbf(x):
    pass
