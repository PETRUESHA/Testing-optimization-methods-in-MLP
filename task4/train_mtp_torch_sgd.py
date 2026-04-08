#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from mpi4py import MPI
import torch
import sys

sys.path.insert(0, "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP")

from mlip_4 import LossFunction, RadialBasisCinf, MTP
from utils import LossBatcher

TRAIN_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/PdAg.json"
VALID_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/validation_PdAg.json"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

train_json_bytes = b""
valid_json_bytes = b""

if rank == 0:
    with open("results_mtp_torch_sgd.csv", "w") as f:
        f.write(
            "pot_num,"
            "train_epa_rmse,train_forces_rmse,"
            "val_epa_rmse,val_forces_rmse,"
            "train_time,steps,epochs,final_loss\n"
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
max_steps = 500
lr = 1e-3
gtol = 1e-6
clip = 1.0
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
    batcher = LossBatcher(train_func, batch_size, True, seed=42 + pot_num)

    x = torch.tensor(pot.params.copy(), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([x], lr=lr)

    g_buf = np.zeros_like(pot.params)

    steps = 0
    epochs = 0
    final_loss = float("nan")

    comm.Barrier()
    start_timer = time.perf_counter()

    stop_training = False
    while steps < max_steps and not stop_training:
        if full_batch:
            batches = [train_func]
        else:
            batcher.new_epoch(pot)
            batches = batcher

        for loss_batch in batches:
            opt.zero_grad()

            x_np = x.detach().cpu().numpy()
            pot.params[:] = x_np

            f = float(loss_batch.calc())
            g_buf.fill(0.0)
            loss_batch.accumulate_grads(g_buf)

            if not np.isfinite(f) or (not np.isfinite(g_buf).all()):
                stop_training = True
                break

            gn = float(np.linalg.norm(g_buf))
            if gn > clip:
                g_step = g_buf * (clip / gn)
                gn = float(np.linalg.norm(g_step))
            else:
                g_step = g_buf.copy()

            x.grad = torch.tensor(g_step, dtype=torch.float64)
            opt.step()

            steps += 1
            final_loss = f

            if gn <= gtol or steps >= max_steps:
                stop_training = True
                break

        epochs += 1

    comm.Barrier()
    train_time = time.perf_counter() - start_timer

    x_final = x.detach().cpu().numpy()
    x_final = comm.bcast(x_final if rank == 0 else None, 0)
    pot.params[:] = x_final

    train_fit_errors = train_func.calc_errors()

    val_func.attach_pot(pot)
    val_func.calc()
    val_fit_errors = val_func.calc_errors()

    train_epa_rmse = float(train_fit_errors.epa())
    train_forces_rmse = float(train_fit_errors.forces())

    val_epa_rmse = float(val_fit_errors.epa())
    val_forces_rmse = float(val_fit_errors.forces())

    if rank == 0:
        with open("results_mtp_torch_sgd.csv", "a") as f:
            f.write(
                f"{pot_num},"
                f"{train_epa_rmse},{train_forces_rmse},"
                f"{val_epa_rmse},{val_forces_rmse},"
                f"{train_time},{steps},{epochs},{final_loss}\n"
            )

        print(f"POT: {pot_num}")
        print("TORCH SGD TRAIN:", train_fit_errors)
        print("TORCH SGD VALID:", val_fit_errors)
        print("TORCH SGD TIME:", train_time, "STEPS:", steps, "EPOCHS:", epochs, "FINAL_LOSS:", final_loss)
        print()
