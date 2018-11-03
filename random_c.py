import numpy as np

class RandomC(object):
    def __init__(self, X, seed=4):
        self.r = np.random
        self.r.seed(seed)
        self.x_min = X.min()
        self.x_max = X.max()
        self.x_mean = X.mean()
        self.x_std = X.std()
        self.x_d = X.shape[0]
        self.c = []

    def product_c(self, k):
        if k % 3 == 0:
            sub_k = k // 3
            self.c = np.hstack((self.uniform_c(sub_k), self.randn_c(sub_k), self.rand_c(sub_k)))
        elif k % 2 == 0:
            sub_k = k // 2
            self.c = np.hstack((self.uniform_c(sub_k), self.randn_c(sub_k)))
        else:
            self.c = self.uniform_c(k)
        return self.c

    def uniform_c(self, sub_k):
        return self.r.uniform(self.x_min, self.x_max, size=(self.x_d, sub_k))

    def randn_c(self, sub_k):
        return self.x_std * self.r.randn(self.x_d, sub_k) + self.x_mean

    def rand_c(self, sub_k):
        return self.r.rand(self.x_d, sub_k) * self.x_mean
