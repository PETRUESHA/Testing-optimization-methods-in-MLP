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

        m_hat = self.m / (1.0 - self.beta1 ** t)
        v_hat = self.v / (1.0 - self.beta2 ** t)

        lr = self.lr_schedule.get_lr(step)
        return x - lr * m_hat / (np.sqrt(v_hat) + self.eps)
