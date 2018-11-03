# encoding: utf-8
import numpy as np
import my_common as mc

class GrandientC(object):
    def __init__(self, c, b, x):
        self.C = c
        self.B = b
        self.X = x
        self.loss = []
        print("degrade for C is ready")

    def degrade(self, lamda, epsion, maxiter=10000):
        _maxiter = maxiter
        print("degrade for C is start")
        while True:
            _maxiter = _maxiter - 1
            delta_C = self.deltaC()
            loss = mc.matrix_distance(lamda * delta_C)
            # print(loss)
            self.loss.append(loss)
            self.C = self.C - lamda * delta_C
            if loss < epsion or _maxiter == 0:
                return self.C, self.loss

    def deltaC(self):
        return 2 * np.dot((np.dot(self.C, self.B) - self.X), self.B.T)

