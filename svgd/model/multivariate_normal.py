import numpy as np
import chainer
import chainer.functions as F
from chainer import Chain, Parameter
from svgd.kernel import rbf


class MVN(Chain):

    def __init__(self, mu, sigma, x0):
        super(MVN, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.n_particle = x0.shape[0]
        with self.init_scope():
            self.theta = Parameter(initializer=x0)

    def logp(self):
        d = self.mu.shape[0]
        mean = np.broadcast_to(self.mu, (self.n_particle, d))
        ln_var = np.broadcast_to(
            2*np.log(self.sigma), (self.n_particle, d))

        logp = -F.gaussian_nll(self.theta, mean, ln_var, reduce='no')
        logp = F.sum(logp, axis=1).reshape(-1)
        logp = F.broadcast_to(logp, (self.n_particle, self.n_particle))
        return logp

    def __call__(self):
        ker = rbf(self.theta.reshape(self.n_particle, -1))
        nlogp = -self.logp()
        loss = F.mean(F.sum(ker.data * nlogp + ker, axis=1))

        chainer.report(
            {
                'loss': loss,
                'nlogp': F.mean(nlogp[0]),
            },
            observer=self,
        )
        return loss
