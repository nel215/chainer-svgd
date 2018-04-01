import chainer.functions as F
from chainer import Chain, Parameter, initializers
from kernel import rbf


class GMM(Chain):

    def __init__(self, n_cluster=2, n_particle=100):
        super(GMM, self).__init__()
        self.n_cluster = n_cluster
        self.n_particle = n_particle
        self._initialized = False

    def _initialize(self, x):
        batch_size, d = x.shape
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

    def logp(self, x):
        if not self._initialized:
            self._initialize(x)

        batch_size, d = x.shape
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

    def __call__(self, x):
        if not self._initialized:
            self._initialize(x)

        ker = 0
        ker += rbf(self.w.reshape(self.n_particle, -1))
        ker += rbf(self.mean.reshape(self.n_particle, -1))
        ker += rbf(self.ln_var.reshape(self.n_particle, -1))
        ker = F.sum(ker, axis=1)
        logp = self.logp(x)
        loss = 0
        for p in range(self.n_particle):
            loss += ker.data[p] * logp[p] + ker[p]
        return -loss
