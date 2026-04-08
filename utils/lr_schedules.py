import numpy as np
from abc import ABC, abstractmethod


# Classes for selecting different learning rates

class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, step: int) -> float:
        ...


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, step: int) -> float:
        return float(self.lr)


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lr0: float = 1e-3, s0: float = 1.0, p: float = 0.5):
        self.lr0 = lr0
        self.s0 = s0
        self.p = p

    def get_lr(self, step: int) -> float:
        return float(self.lr0 * (self.s0 / (self.s0 + step)) ** self.p)
