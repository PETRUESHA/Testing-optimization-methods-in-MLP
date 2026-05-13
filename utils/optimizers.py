import numpy as np
from abc import ABC, abstractmethod
from utils.lr_schedules import LearningRateSchedule


class OptimizerState(ABC):
    @abstractmethod
    def step(self, x: np.ndarray, g: np.ndarray, step: int) -> np.ndarray:
        pass


class SGD(OptimizerState):
    def __init__(self, lr_schedule: LearningRateSchedule):
        self.lr_schedule = lr_schedule

    def step(self, x: np.ndarray, g: np.ndarray, step: int) -> np.ndarray:
        lr = self.lr_schedule.get_lr(step)
        return x - lr * g


class Momentum(OptimizerState):
    def __init__(self, lr_schedule: LearningRateSchedule, beta: float = 0.9):
        self.lr_schedule = lr_schedule
        self.beta = beta
        self.v = None

    def step(self, x: np.ndarray, g: np.ndarray, step: int) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(x)
        lr = self.lr_schedule.get_lr(step)
        self.v = self.beta * self.v + (1.0 - self.beta) * g
        return x - lr * self.v


class Adam(OptimizerState):
    def __init__(
        self,
        lr_schedule: LearningRateSchedule,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr_schedule = lr_schedule
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def step(self, x: np.ndarray, g: np.ndarray, step: int) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        t = step + 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)

        m_hat = self.m / (1.0 - self.beta1**t)
        v_hat = self.v / (1.0 - self.beta2**t)

        lr = self.lr_schedule.get_lr(step)
        return x - lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _newton_schulz5(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    if G.ndim != 2:
        raise ValueError("Newton–Schulz expects a 2D matrix.")

    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.astype(np.float64, copy=False)
    nrm = np.linalg.norm(X)
    X = X / (nrm + eps)

    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)

    if transposed:
        X = X.T

    return X


def pick_shape_factorization(N: int) -> tuple[int, int, int]:
    if N <= 0:
        raise ValueError("N must be positive.")
    m = int(np.floor(np.sqrt(N)))
    while m > 1 and (N % m) != 0:
        m -= 1
    n = N // m
    return m, n, 0


def pick_shape_pad_sqrt(N: int) -> tuple[int, int, int]:
    if N <= 0:
        raise ValueError("N must be positive.")
    m = int(np.round(np.sqrt(N)))
    m = max(1, m)
    n = int(np.ceil(N / m))
    pad = m * n - N
    return m, n, pad


class Muon(OptimizerState):
    """
    Parameters:
      lr_schedule : LearningRateSchedule
          Defines lr(step).

      beta : float
          Momentum EMA coefficient. Internal buffer:
            m = beta*m + (1-beta)*g

      ns_steps : int
          Number of Newton–Schulz iterations for orthogonalization.

      nesterov : bool
          If True use Nesterov-like combination:
            u = (1-beta)*g + beta*m
          else:
            u = m

      eps : float
          Small constant for numerical stability in normalization inside
          Newton–Schulz.

      shape_mode : str
          How to choose (m,n) when param_shape is None:
            "pad_sqrt": near-square with zero padding
            "factor": exact divisor search (no padding)

      param_shape : tuple[int,int] | None
          If set, overrides shape_mode and forces this (m,n).
          If m*n > N => pads zeros; if m*n < N => error.
    """

    def __init__(
        self,
        lr_schedule: LearningRateSchedule,
        beta: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        eps: float = 1e-7,
        shape_mode: str = "pad_sqrt",  # "pad_sqrt" | "factor"
        param_shape: tuple[int, int] | None = None
    ):
        self.lr_schedule = lr_schedule
        self.beta = beta
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.eps = eps
        self.shape_mode = shape_mode
        self.param_shape = param_shape

        self.m_buf = None
        self._mn_pad = None  # (m, n, pad)

    def _ensure_shape(self, N: int) -> tuple[int, int, int]:
        if self.param_shape is not None:
            m, n = self.param_shape
            if m <= 0 or n <= 0:
                raise ValueError("param_shape must be positive.")
            pad = m * n - N
            if pad < 0:
                raise ValueError("param_shape too small for vector length.")
            return m, n, pad

        if self._mn_pad is None:
            if self.shape_mode == "pad_sqrt":
                self._mn_pad = pick_shape_pad_sqrt(N)
            elif self.shape_mode == "factor":
                self._mn_pad = pick_shape_factorization(N)
            else:
                raise ValueError("shape_mode must be 'pad_sqrt' or 'factor'.")
        return self._mn_pad

    def step(self, x: np.ndarray, g: np.ndarray, step: int) -> np.ndarray:
        N = x.size
        if g.size != N:
            raise ValueError("x and g must have the same size.")

        if self.m_buf is None:
            self.m_buf = np.zeros_like(x)

        self.m_buf = self.beta * self.m_buf + (1.0 - self.beta) * g

        if self.nesterov:
            u = (1.0 - self.beta) * g + self.beta * self.m_buf
        else:
            u = self.m_buf

        m, n, pad = self._ensure_shape(N)

        if pad > 0:
            u2 = np.concatenate([u, np.zeros(pad, dtype=u.dtype)], axis=0)
        else:
            u2 = u

        U = u2.reshape(m, n)
        U = _newton_schulz5(U, steps=self.ns_steps, eps=self.eps)

        scale = float(np.sqrt(max(1.0, m / n)))
        U = U * scale

        u_out = U.reshape(-1)
        if pad > 0:
            u_out = u_out[:N]

        lr = self.lr_schedule.get_lr(step)
        return x - lr * u_out
