#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from mpi4py import MPI
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP")

from mlip_4 import LossFunction, RadialBasisCinf, MTP
import utils
from utils import MlipWrapper

TRAIN_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/PdAg.json"
VALID_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/validation_PdAg.json"
RESULT_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/task4/results/results_mtp_adam_then_lbfgs.csv"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

train_json_bytes = b""
valid_json_bytes = b""

if rank == 0:
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        f.write(
            "pot_num,"
            "train_epa_rmse,train_forces_rmse,"
            "val_epa_rmse,val_forces_rmse,"
            "time_adam,steps_adam,epochs_adam,final_loss_adam,"
            "time_lbfgs,nit,success,final_loss_lbfgs\n"
        )

    with open(TRAIN_PATH, "rb") as f:
        train_json_bytes = f.read()
    with open(VALID_PATH, "rb") as f:
        valid_json_bytes = f.read()

train_json_bytes = comm.bcast(train_json_bytes, 0)
valid_json_bytes = comm.bcast(valid_json_bytes, 0)

rb = RadialBasisCinf(size=8, min_dist=1.9, cutoff=5.0)
species_order = [0, 1]
level = 16
jit = True

batch_size = 32
adam_steps = 800
lbfgs_maxiter = 300
adam_lr = 1e-3
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
gtol = 1e-6
clip_adam = 1.0
full_batch = False

for pot_num in range(1, 6):
    pot_json = b""

    if rank == 0:
        pot = MTP(radial_basis=rb, species=species_order, level=level, jit=jit)
        np.random.seed(42 + pot_num)
        pot.params[:] = np.random.uniform(low=-0.1, high=0.1, size=len(pot.params))
        pot_json = pot.to_json_bytes()

    pot_json = comm.bcast(pot_json, 0)
    pot = MTP.from_json_bytes(pot_json)

    train_func = LossFunction.from_json_bytes(train_json_bytes, True)
    val_func = LossFunction.from_json_bytes(valid_json_bytes, True)

    train_func.attach_pot(pot)
    batcher = utils.LossBatcher(train_func, batch_size, True, seed=42 + pot_num)

    lr_schedule = utils.lr_schedules.ConstantLR(adam_lr)
    opt = utils.optimizers.Adam(lr_schedule=lr_schedule, beta1=beta1, beta2=beta2, eps=eps)

    comm.Barrier()
    t0 = time.perf_counter()

    trainer = utils.mlip_trainer.MlipTrainer(gtol=gtol, max_steps=adam_steps)
    hist = trainer.train(
        pot=pot,
        train_func=train_func,
        optimizer=opt,
        batcher=batcher,
        full_batch=full_batch,
        clip=clip_adam,
    )

    comm.Barrier()
    time_adam = time.perf_counter() - t0

    final_loss_adam = float(hist.loss[-1]) if len(hist.loss) else float("nan")
    steps_adam = int(hist.steps)
    epochs_adam = int(hist.epochs)

    wrapper = MlipWrapper(train_func, pot)
    x0 = pot.params.copy()

    comm.Barrier()
    t1 = time.perf_counter()

    res = minimize(
        fun=wrapper.value,
        x0=x0,
        jac=wrapper.grad,
        method="L-BFGS-B",
        options={"maxiter": lbfgs_maxiter, "gtol": gtol, "disp": False},
    )

    comm.Barrier()
    time_lbfgs = time.perf_counter() - t1

    pot.params[:] = res.x

    train_fit_errors = train_func.calc_errors()

    val_func.attach_pot(pot)
    val_func.calc()
    val_fit_errors = val_func.calc_errors()

    train_epa_rmse = float(train_fit_errors.epa())
    train_forces_rmse = float(train_fit_errors.forces())
    val_epa_rmse = float(val_fit_errors.epa())
    val_forces_rmse = float(val_fit_errors.forces())

    nit = int(getattr(res, "nit", -1))
    success = int(bool(getattr(res, "success", False)))
    final_loss_lbfgs = float(getattr(res, "fun", float("nan")))

    if rank == 0:
        with open(RESULT_PATH, "a") as f:
            f.write(
                f"{pot_num},"
                f"{train_epa_rmse},{train_forces_rmse},"
                f"{val_epa_rmse},{val_forces_rmse},"
                f"{time_adam},{steps_adam},{epochs_adam},{final_loss_adam},"
                f"{time_lbfgs},{nit},{success},{final_loss_lbfgs}\n"
            )

        print(f"POT: {pot_num}")
        print("ADAM->LBFGS TRAIN:", train_fit_errors)
        print("ADAM->LBFGS VALID:", val_fit_errors)
        print("ADAM:", time_adam, "STEPS:", steps_adam, "EPOCHS:", epochs_adam, "FINAL_LOSS:", final_loss_adam)
        print("LBFGS:", time_lbfgs, "NIT:", nit, "SUCCESS:", bool(success), "FINAL_LOSS:", final_loss_lbfgs)
        print()
