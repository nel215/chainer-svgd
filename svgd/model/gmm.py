import chainer
import numpy as np
import chainer.functions as F
from chainer import Chain, Parameter
from svgd.kernel import rbf


class GMM(Chain):

    def __init__(self, mu, sigma, x0):
        '''
        x0: (k, n_particle, d)
        '''
        super(GMM, self).__init__()
        self.mu = mu
        self.sigma = sigma
        with self.init_scope():
            self.theta = Parameter(initializer=x0)

    def logp(self):
        res = 0
        n_particle = self.theta.shape[0]
        n_cluster = self.mu.shape[0]
        d = self.mu.shape[1]
        for k in range(n_cluster):
            mean = F.broadcast_to(self.mu[k], (n_particle, d))
            ln_var = F.broadcast_to(
                2*np.log(self.sigma[k]), (n_particle, d),
            )
            theta = self.theta[:, k, :]
            p = -F.gaussian_nll(theta, mean, ln_var, 'no')
            p = F.exp(F.sum(p, axis=1))
            res += 0.5 * p

        res = F.log(res)
        res = F.broadcast_to(res, (n_particle, n_particle))
        return res

    def __call__(self):
        n_particle = self.theta.shape[0]
        ker = 0
        for k in range(self.theta.shape[1]):
            ker += rbf(self.theta[:, k, :].reshape(n_particle, -1))
        nlogp = -self.logp()
        loss = F.mean(F.sum(ker.data * nlogp + ker, axis=1))
        # loss = F.mean(F.sum(nlogp, axis=1))
        chainer.report(
            {
                'loss': loss,
                'nlogp': F.mean(nlogp[0]),
            },
            observer=self,
        )
        return loss
