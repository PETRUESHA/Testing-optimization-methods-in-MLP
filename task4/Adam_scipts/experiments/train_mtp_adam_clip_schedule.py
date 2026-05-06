#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from mpi4py import MPI
import sys
import json
import os

sys.path.insert(0, "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP")

from mlip_4 import LossFunction, RadialBasisCinf, MTP
import utils

TRAIN_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/PdAg.json"
VALID_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/validation_PdAg.json"
RESULT_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/task4/results/results_mtp_adam_clip_schedule.csv"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

train_json_bytes = b""
valid_json_bytes = b""


def merge_histories(h1, h2):
    h = type(h1)()
    h.loss = list(h1.loss) + list(h2.loss)
    h.grad_norm = list(h1.grad_norm) + list(h2.grad_norm)
    h.lr = list(h1.lr) + list(h2.lr)
    h.steps = int(h1.steps) + int(h2.steps)
    h.epochs = int(h1.epochs) + int(h2.epochs)
    return h


def write_row_with_history(
    filename,
    pot_num,
    train_epa_rmse,
    train_forces_rmse,
    val_epa_rmse,
    val_forces_rmse,
    train_time,
    steps_done,
    epochs_done,
    final_loss,
    hist,
    clip_warmup,
    warmup_steps,
    clip_main,
    total_steps,
):
    losses = json.dumps([float(x) for x in hist.loss])
    grad_norms = json.dumps([float(x) for x in hist.grad_norm])
    lrs = json.dumps([float(x) for x in hist.lr])

    with open(filename, "a") as f:
        f.write(
            f"{pot_num},"
            f"{clip_warmup},{warmup_steps},{clip_main},{total_steps},"
            f"{train_epa_rmse},{train_forces_rmse},"
            f"{val_epa_rmse},{val_forces_rmse},"
            f"{train_time},{steps_done},{epochs_done},{final_loss},"
            f"\"{losses}\",\"{grad_norms}\",\"{lrs}\"\n"
        )


if rank == 0:
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        f.write(
            "pot_num,"
            "clip_warmup,warmup_steps,clip_main,total_steps,"
            "train_epa_rmse,train_forces_rmse,"
            "val_epa_rmse,val_forces_rmse,"
            "train_time,steps,epochs,final_loss,"
            "losses,grad_norms,lrs\n"
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
max_steps = 1500
warmup_steps = 200
lr = 1e-3
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
gtol = 1e-6
clip_warmup = 1.0
clip_main = 10.0
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

    lr_schedule = utils.lr_schedules.ConstantLR(lr)
    opt = utils.optimizers.Adam(lr_schedule=lr_schedule, beta1=beta1, beta2=beta2, eps=eps)

    comm.Barrier()
    start_timer = time.perf_counter()

    trainer_warmup = utils.mlip_trainer.MlipTrainer(gtol=gtol, max_steps=warmup_steps)
    hist1 = trainer_warmup.train(
        pot=pot,
        train_func=train_func,
        optimizer=opt,
        batcher=batcher,
        full_batch=full_batch,
        clip=clip_warmup,
    )

    remaining_steps = max_steps - int(hist1.steps)
    if remaining_steps > 0:
        trainer_main = utils.mlip_trainer.MlipTrainer(gtol=gtol, max_steps=remaining_steps)
        hist2 = trainer_main.train(
            pot=pot,
            train_func=train_func,
            optimizer=opt,
            batcher=batcher,
            full_batch=full_batch,
            clip=clip_main,
        )
        hist = merge_histories(hist1, hist2)
    else:
        hist = hist1

    comm.Barrier()
    train_time = time.perf_counter() - start_timer

    train_fit_errors = train_func.calc_errors()

    val_func.attach_pot(pot)
    val_func.calc()
    val_fit_errors = val_func.calc_errors()

    train_epa_rmse = float(train_fit_errors.epa())
    train_forces_rmse = float(train_fit_errors.forces())
    val_epa_rmse = float(val_fit_errors.epa())
    val_forces_rmse = float(val_fit_errors.forces())

    final_loss = float(hist.loss[-1]) if len(hist.loss) else float("nan")
    steps_done = int(hist.steps)
    epochs_done = int(hist.epochs)

    if rank == 0:
        write_row_with_history(
            RESULT_PATH,
            pot_num,
            train_epa_rmse,
            train_forces_rmse,
            val_epa_rmse,
            val_forces_rmse,
            train_time,
            steps_done,
            epochs_done,
            final_loss,
            hist,
            clip_warmup,
            warmup_steps,
            clip_main,
            max_steps,
        )

        print(f"POT: {pot_num}")
        print("ADAM CLIP-SCHEDULE TRAIN:", train_fit_errors)
        print("ADAM CLIP-SCHEDULE VALID:", val_fit_errors)
        print("TIME:", train_time, "STEPS:", steps_done, "EPOCHS:", epochs_done, "FINAL_LOSS:", final_loss)
        print()
