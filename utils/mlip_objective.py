import numpy as np

# Class for calculating gradients of the given LossFunction
# Normalazing gradient, if clip parameter is True


class MlipObjective:
    def __init__(self, pot, clip: float | None = None):
        self.pot = pot
        self.clip = clip
        self.g_buf = np.zeros_like(pot.params)

    def loss_and_grad(self, loss_func, x: np.ndarray):
        self.pot.params[:] = x
        f = float(loss_func.calc())

        self.g_buf.fill(0.0)
        loss_func.accumulate_grads(self.g_buf)

        if not np.isfinite(f) or (not np.isfinite(self.g_buf).all()):
            return None, None, None

        g = self.g_buf.copy()
        gn = float(np.linalg.norm(g))

        if self.clip is not None and gn > self.clip:
            g = g * (self.clip / gn)
            gn = float(np.linalg.norm(g))

        return f, g, gn
