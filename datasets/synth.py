import torch
import numpy as np
from scipy.stats import truncnorm
from torch.utils.data import Dataset

class oneD_synth(Dataset):
    def __init__(self, x_min=-1.0, x_max=1.0, size=int(1e5), distr='eq',
                 mean=None, std=None, custom_x=None, noise=False):
        """
        :param x_min:
        :param x_max:
        :param size: total num datapoints
        :param distr: one of 'eq', 'unif', 'normal', 'custom'
        """
        self.size = size
        self.noise = noise
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

    def gaussian_noise(self, X, std_1=1.0, std_2=5.0, std_3=0.5):
        if not self.noise:
            return 0

        # q_1, q_2, q_3 = np.quantile([np.min(self.x), np.max(self.x)], [0.25,0.5,0.75])
        # self.q_1, self.q_2, self.q_3 = q_1, q_2, q_3
        # if (X < q_1) or (X > q_3):
        #     return np.random.normal(0, std_1)
        # elif (q_1 < X) and (X < q_2):
        #     return np.random.normal(0, std_2)
        # elif (q_2 < X) and (X < q_3):
        #     return np.random.normal(0, std_3)

        q_1 = np.quantile([np.min(self.x), np.max(self.x)], [0.5])
        self.q_1 = q_1
        if X < self.q_1:
            return np.random.normal(0, std_1)
        elif self.q_1 < X:
            return np.random.normal(0, std_2)

    def oneD_fn(self, X):
        raise NotImplementedError("Must implement 1D function")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X = self.x[idx]
        y = self.oneD_fn(X) + self.gaussian_noise(X)

        return X, y

class oneD_sine(oneD_synth):
    def oneD_fn(self, X):
        return np.sin(X)

class oneD_cubed(oneD_synth):
    def oneD_fn(self, X):
        return np.power(X,3)

class oneD_linear(oneD_synth):
    def __init__(self, x_min=-1.0, x_max=1.0, size=int(1e5), distr='eq',
                 mean=None, std=None, custom_x=None, noise=False,
                 lin=1, const=0):

        super().__init__(x_min, x_max, size, distr, mean, std, custom_x, noise)
        self.lin, self.const = lin, const

    def oneD_fn(self, X):
        return self.lin*X + self.const

class oneD_parabola(oneD_synth):
    def __init__(self, x_min=-1.0, x_max=1.0, size=int(1e5), distr='eq',
                 mean=None, std=None, custom_x=None, noise=False,
                 quad=3, lin=-5, const=8):

        super().__init__(x_min, x_max, size, distr, mean, std, custom_x, noise)
        self.quad, self.lin, self.const = quad, lin, const

    def oneD_fn(self, X):
        return self.quad*X**2 + self.lin*X + self.const

class oneD_test_1(oneD_synth):
    def oneD_fn(self, X):
        return X*np.sin(X) + 0.3*np.random.normal(0, 1) + 0.3*X*np.random.normal(0, 1)

if __name__=='__main__':
    # test_fn = oneD_test_1(size=1000, noise=True, distr='unif')
    test_fn = oneD_test_1(size=1000, noise=False, distr='unif',
                          x_min=-4, x_max=14)
    batch_size=test_fn.__len__()
    dataloader = torch.utils.data.DataLoader(dataset=test_fn, batch_size=batch_size)
    X, y = next(iter(dataloader))

    import matplotlib.pyplot as plt
    X_np = X.numpy()
    order = np.argsort(X_np.flatten(), axis=0)
    X_np = X_np[order]
    y_np = y.numpy()[order]
    import pdb; pdb.set_trace()
    plt.plot(X_np, test_fn.oneD_fn(X_np), '--', c='r')
    plt.plot(X_np, y_np, '.', markersize=1)
    # plt.axvline(test_fn.q_1, c='k')
    # plt.axvline(test_fn.q_2, c='k')
    # plt.axvline(test_fn.q_3, c='k')
    plt.show()
