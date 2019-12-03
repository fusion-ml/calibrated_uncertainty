import numpy as np
from scipy.stats import truncnorm
from torch.utils.data import Dataset

class oneD_synth(Dataset):
    def __init__(self, x_min=-1.0, x_max=1.0, size=int(1e5), distr='eq', mean=None, std=None, custom_x=None):
        """
        :param x_min:
        :param x_max:
        :param size: total num datapoints
        :param distr: one of 'eq', 'unif', 'normal', 'custom'
        """
        self.size = size
        if distr=='eq':
            self.x = np.linspace(x_min, x_max, size).reshape(-1,1)
        elif distr=='unif':
            self.x = np.random.uniform(x_min, x_max, size).reshape(-1,1)
        elif distr=='norm':
            if (x_min is not None) and (x_max is not None):
                x_distr = truncnorm((x_min-mean)/std, (x_max-mean)/std,
                                    loc=mean, scale=sigma)
                self.x = x_distr.rvs(size).reshape(-1,1)
            else:
                x_distr = np.random.normal(loc=mean, scale=std, size=size)
        elif distr=='custom':
            assert(custom_x is not None)
            self.x = np.array(custom_x).reshape(-1,1)

    def oneD_fn(self, X):
        raise NotImplementedError("Must implement 1D function")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X = self.x[idx]
        y = self.oneD_fn(X)

        return X, y

class oneD_sine(oneD_synth):
    def oneD_fn(self, X):
        return np.sin(X)

class oneD_cubed(oneD_synth):
    def oneD_fn(self, X):
        return np.power(X,3)

class oneD_linear(oneD_synth):
    def __init__(self, x_min=-1.0, x_max=1.0, size=int(1e5), distr='eq',
                 mean=None, std=None, custom_x=None,
                 lin=1, const=0):

        super().__init__(x_min, x_max, size, distr, mean, std, custom_x)
        self.lin, self.const = lin, const

    def oneD_fn(self, X):
        return self.lin*X + self.const

class oneD_parabola(oneD_synth):
    def __init__(self, x_min=-1.0, x_max=1.0, size=int(1e5), distr='eq',
                 mean=None, std=None, custom_x=None,
                 quad=3, lin=-5, const=8):

        super().__init__(x_min, x_max, size, distr, mean, std, custom_x)
        self.quad, self.lin, self.const = quad, lin, const

    def oneD_fn(self, X):
        return self.quad*X**2 + self.lin*X + self.const

