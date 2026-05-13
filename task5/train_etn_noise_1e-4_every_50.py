#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import os
from mpi4py import MPI
from mlip_4 import LossFunction, RadialBasisCinf, ETN, Trainer

TRAIN_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/MoNbTaW/MoNbTaW_train.json"
VALID_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/MoNbTaW/MoNbTaW_valid.json"
RESULT_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/task5/results/results_etn_MoNbTaw.csv"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

train_json_bytes = b""
valid_json_bytes = b""

if rank == 0:

    with open(RESULT_PATH, "w") as f:
        f.write(
            "pot_num,noise_sigma,noise_every,"
            "train_energy_rmse,train_energy_atom_rmse,train_forces_rmse,"
            "val_energy_rmse,val_energy_atom_rmse,val_forces_rmse,"
            "train_time\n"
        )

    with open(TRAIN_PATH, "rb") as f:
        train_json_bytes = f.read()

    with open(VALID_PATH, "rb") as f:
        valid_json_bytes = f.read()

train_json_bytes = comm.bcast(train_json_bytes, 0)
valid_json_bytes = comm.bcast(valid_json_bytes, 0)

rb = RadialBasisCinf(size=8, min_dist=1.9, cutoff=5.0)
species_order = [0, 1, 2, 3]
etn_shape = [[1], [6], [3]]
ett_ranks = [[1], [1]]
jit = True

max_iter_count = 500
noise_every = 50
noise_sigma = 1e-4


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
        pot.params[:] = np.random.uniform(low=-1.0, high=1.0, size=len(pot.params))
        pot_json = pot.to_json_bytes()

        print(f"\n=== POT {pot_num} ===")
        print("init params head:", pot.params[:5])

    pot_json = comm.bcast(pot_json, 0)
    pot = ETN.from_json_bytes(pot_json)

    train_func = LossFunction.from_json_bytes(train_json_bytes, True)
    val_func = LossFunction.from_json_bytes(valid_json_bytes, True)

    train_func.attach_pot(pot)

    comm.Barrier()
    loss_init = float(train_func.calc())
    comm.Barrier()

    if rank == 0:
        print("init loss:", loss_init)

    steps_done = 0

    comm.Barrier()
    start_timer = time.perf_counter()

    while steps_done < max_iter_count:
        cur_steps = min(noise_every, max_iter_count - steps_done)

        trainer = Trainer(train_func)
        trainer.train(pot, max_iter_count=cur_steps)

        steps_done += cur_steps

        if steps_done < max_iter_count:
            if rank == 0:
                np.random.seed(100000 + 1000 * pot_num + steps_done)
                pot.params[:] += noise_sigma * np.random.randn(len(pot.params))
                pot_json = pot.to_json_bytes()

            pot_json = comm.bcast(pot_json, 0)
            pot = ETN.from_json_bytes(pot_json)

            train_func.attach_pot(pot)

    comm.Barrier()
    train_time = time.perf_counter() - start_timer

    comm.Barrier()
    loss_final = float(train_func.calc())
    comm.Barrier()

    if rank == 0:
        print("final params head:", pot.params[:5])
        print("final loss:", loss_final)

    train_fit_errors = train_func.calc_errors()

    val_func.attach_pot(pot)
    val_func.calc()
    val_fit_errors = val_func.calc_errors()

    train_energy_rmse = float(train_fit_errors.energy())
    train_epa_rmse = float(train_fit_errors.epa())
    train_forces_rmse = float(train_fit_errors.forces())

    val_energy_rmse = float(val_fit_errors.energy())
    val_epa_rmse = float(val_fit_errors.epa())
    val_forces_rmse = float(val_fit_errors.forces())

    if rank == 0:
        with open(RESULT_PATH, "a") as f:
            f.write(
                f"{pot_num},{noise_sigma},{noise_every},"
                f"{train_energy_rmse},{train_epa_rmse},{train_forces_rmse},"
                f"{val_energy_rmse},{val_epa_rmse},{val_forces_rmse},"
                f"{train_time}\n"
            )

        print("TRAIN:", train_fit_errors)
        print("VALID:", val_fit_errors)
        print("TRAIN_TIME:", train_time)
