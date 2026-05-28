#!/usr/bin/env python
# coding: utf-8


import sys
import os
sys.path.insert(
    0, "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP")
import utils
import json
from mlip_4 import LossFunction, RadialBasisCinf, ETN
import numpy as np
import time
from mpi4py import MPI

TRAIN_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/PdAg.json"
VALID_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/validation_PdAg.json"
RESULT_PATH = "/Users/peterzarenkov/projects/course_work/Testing-optimization-methods-in-MLP/all_etn_5000/results/AgPd_results_etn_my_muon_factorization.csv"

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
etn_shape = [[1], [6], [3]]
ett_ranks = [[1], [1]]
jit = True

batch_size = 32
max_steps = 5000
lr = 1e-3
gtol = 1e-6
clip = 1.0
full_batch = False

muon_beta = 0.95
muon_ns_steps = 5
muon_nesterov = True
muon_eps = 1e-7
muon_shape_mode = "factor"
muon_param_shape = None

for pot_num in range(1, 6):
    pot_json = b""

    if rank == 0:
        pot = ETN(
            radial_basis=rb,
            etn_shape=etn_shape,
            species=species_order,
            ett_ranks=ett_ranks,
            jit=jit,
        )
        np.random.seed(42 + pot_num)
        pot.params[:] = np.random.uniform(
            low=-0.001, high=0.001, size=len(pot.params))
        pot_json = pot.to_json_bytes()

    pot_json = comm.bcast(pot_json, 0)
    pot = ETN.from_json_bytes(pot_json)

    train_func = LossFunction.from_json_bytes(train_json_bytes, True)
    val_func = LossFunction.from_json_bytes(valid_json_bytes, True)

    train_func.attach_pot(pot)
    batcher = utils.LossBatcher(
        train_func, batch_size, True, seed=42 + pot_num)

    lr_schedule = utils.lr_schedules.ConstantLR(lr)

    opt = utils.optimizers.Muon(
        lr_schedule=lr_schedule,
        beta=muon_beta,
        ns_steps=muon_ns_steps,
        nesterov=muon_nesterov,
        eps=muon_eps,
        shape_mode=muon_shape_mode,
        param_shape=muon_param_shape,
    )

    trainer = utils.mlip_trainer.MlipTrainer(gtol=gtol, max_steps=max_steps)

    comm.Barrier()
    start_timer = time.perf_counter()

    hist = trainer.train(
        pot=pot,
        train_func=train_func,
        optimizer=opt,
        batcher=batcher,
        full_batch=full_batch,
        clip=clip,
    )

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

    losses = json.dumps([float(x) for x in hist.loss])
    grad_norms = json.dumps([float(x) for x in hist.grad_norm])
    lrs = json.dumps([float(x) for x in hist.lr])

    if rank == 0:
        with open(RESULT_PATH, "a") as f:
            f.write(
                f"{pot_num},"
                f"{train_epa_rmse},{train_forces_rmse},"
                f"{val_epa_rmse},{val_forces_rmse},"
                f"{train_time},{steps_done},{epochs_done},{final_loss},"
                f"\"{losses}\",\"{grad_norms}\",\"{lrs}\"\n"
            )

        print(f"POT: {pot_num}")
        print("MY MUON TRAIN:", train_fit_errors)
        print("MY MUON VALID:", val_fit_errors)
        print("MY MUON TIME:", train_time, "STEPS:", steps_done,
              "EPOCHS:", epochs_done, "FINAL_LOSS:", final_loss)
        print()
