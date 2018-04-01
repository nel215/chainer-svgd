import chainer.functions as F
from chainer import Chain, Parameter, initializers


class GMM(Chain):

    def __init__(self, n_cluster=2, n_particle=100):
        super(GMM, self).__init__()
        self.n_cluster = n_cluster
        self.n_particle = n_particle
        self._initialized = False

    def logp(self, x):
        batch_size, d = x.shape
        if not self._initialized:
            with self.init_scope():
                self.w = Parameter(
                    initializers.Normal(),
                    shape=(self.n_particle, self.n_cluster))
                self.mean = Parameter(
                    initializers.Normal(),
                    shape=(self.n_particle, self.n_cluster, d))
                self.ln_var = Parameter(
                    initializers.Normal(),
                    shape=(self.n_particle, self.n_cluster, d))
            self._initialized = True

        res = []
        for p in range(self.n_particle):
            logp = 0
            for k in range(self.n_cluster):
                w = self.w[p][k]
                w = F.sigmoid(w)
                mean = self.mean[p][k]
                mean = F.broadcast_to(mean, (batch_size, d))
                ln_var = self.ln_var[p][k]
                ln_var = F.sigmoid(ln_var)
                ln_var = F.broadcast_to(ln_var, (batch_size, d))
                logp += F.log(w) - F.gaussian_nll(x, mean, ln_var)
            res.append(logp)

        return res

    def kernel(self, theta):
        pass

    def __call__(self, x):
        pass
