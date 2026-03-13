import numpy as np


class MlipWrapper:
    def __init__(self, func, pot):
        self.func = func
        self.pot = pot
        self.g = np.zeros_like(pot.params)
        self._x_last = None
        self._f_last = None
        self._g_last = None

    def set_x(self, x):
        self.pot.params[:] = x

    def value(self, x):
        if self._x_last is not None and np.array_equal(x, self._x_last):
            return self._f_last
        self.set_x(x)
        f = float(self.func.calc())
        self._x_last = x.copy()
        self._f_last = f
        self._g_last = None
        return f

    def grad(self, x):
        if self._x_last is not None and np.array_equal(x, self._x_last) and self._g_last is not None:
            return self._g_last.copy()
        self.set_x(x)
        self.g.fill(0.0)
        self.func.accumulate_grads(self.g)
        g = self.g.copy()
        self._x_last = x.copy()
        self._g_last = g
        self._f_last = None
        return g
