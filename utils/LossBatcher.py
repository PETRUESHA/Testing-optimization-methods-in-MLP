import numpy as np
from mlip_4 import LossFunction

# Spliting giving LossFunction to batches and returning iterator of the list with LossFunctions, containing batches

class LossBatcher:
    def __init__(self, full_func, batch_size, shuffle=True, seed=0, drop_last=False):
        self.terms = list(full_func)
        self.n = len(self.terms)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.batch_funcs = []

    def new_epoch(self, pot):
        idx = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(idx)

        self.batch_funcs = []
        end = self.n - (self.n % self.batch_size) if self.drop_last else self.n

        for start in range(0, end, self.batch_size):
            bf = LossFunction()
            for j in idx[start:start + self.batch_size]:
                bf.add(self.terms[int(j)])
            bf.attach_pot(pot)
            self.batch_funcs.append(bf)

    def __iter__(self):
        return iter(self.batch_funcs)

    def __len__(self):
        return len(self.batch_funcs)
