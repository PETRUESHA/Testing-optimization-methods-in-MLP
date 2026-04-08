from utils.mlip_objective import MlipObjective
from utils.optimizers import OptimizerState


# Helping class, obtaining history of training
# Containing losses, gradient norms and learning rates of each step of training

class TrainHistory:
    def __init__(self):
        self.loss = []
        self.grad_norm = []
        self.lr = []
        self.steps = 0
        self.epochs = 0

    def record(self, f, gn, lr_val):
        self.loss.append(float(f))
        self.grad_norm.append(float(gn))
        self.lr.append(float(lr_val))

# Main training class, where main cycle is placed and other classes are used

class MlipTrainer:
    def __init__(self, gtol: float = 1e-6, max_steps: int = 500):
        self.gtol = gtol
        self.max_steps = max_steps

    def train(
        self,
        pot,
        train_func,
        optimizer: OptimizerState,
        batcher,
        full_batch: bool = False,
        clip: float | None = None,
    ):
        train_func.attach_pot(pot)
        obj = MlipObjective(pot, clip=clip)

        x = pot.params.copy()
        hist = TrainHistory()

        step = 0
        epoch = 0

        while hist.steps < self.max_steps:
            if full_batch:
                batches = [train_func]
            else:
                batcher.new_epoch(pot)
                batches = batcher

            for loss_batch in batches:
                f, g, gn = obj.loss_and_grad(loss_batch, x)
                if f is None:
                    pot.params[:] = x
                    hist.epochs = epoch
                    return hist

                lr_val = optimizer.lr_schedule.get_lr(step) if hasattr(optimizer, "lr_schedule") else float("nan")
                hist.record(f, gn, lr_val)

                x = optimizer.step(x, g, step)

                hist.steps += 1
                step += 1

                if gn <= self.gtol or hist.steps >= self.max_steps:
                    pot.params[:] = x
                    hist.epochs = epoch
                    return hist

            epoch += 1
            hist.epochs = epoch

        pot.params[:] = x
        return hist
