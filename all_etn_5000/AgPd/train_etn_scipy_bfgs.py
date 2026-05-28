#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import os
from mpi4py import MPI
from scipy.optimize import minimize
import sys

sys.path.insert(0, "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP")

from mlip_4 import LossFunction, RadialBasisCinf, ETN
from utils import MlipWrapper


TRAIN_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/PdAg.json"
VALID_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/validation_PdAg.json"
RESULT_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/all_etn_5000/results/AgPd_results_etn_scipy_bfgs.csv"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

train_json_bytes = b""
valid_json_bytes = b""

if rank == 0:
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    with open(RESULT_PATH, "w") as f:
        f.write(
            "pot_num,"
            "train_energy_atom_rmse,train_forces_rmse,"
            "val_energy_atom_rmse,val_forces_rmse,"
            "train_time,nit,success,final_loss\n"
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

max_iter_count = 5000
gtol = 1e-6

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
        pot.params[:] = np.random.uniform(low=-0.001, high=0.001, size=len(pot.params))

        pot_json = pot.to_json_bytes()

    pot_json = comm.bcast(pot_json, 0)
    pot = ETN.from_json_bytes(pot_json)

    train_func = LossFunction.from_json_bytes(train_json_bytes, True)
    val_func = LossFunction.from_json_bytes(valid_json_bytes, True)

    train_func.attach_pot(pot)
    wrapper = MlipWrapper(train_func, pot)

    x0 = pot.params.copy()

    comm.Barrier()
    start_timer = time.perf_counter()

    res = minimize(
        fun=wrapper.value,
        x0=x0,
        jac=wrapper.grad,
        method="BFGS",
        options={"maxiter": max_iter_count, "gtol": gtol, "disp": False},
    )

    comm.Barrier()
    train_time = time.perf_counter() - start_timer

    pot.params[:] = res.x

    train_fit_errors = train_func.calc_errors()

    val_func.attach_pot(pot)
    val_func.calc()
    val_fit_errors = val_func.calc_errors()

    train_epa_rmse = float(train_fit_errors.epa())
    train_forces_rmse = float(train_fit_errors.forces())

    val_epa_rmse = float(val_fit_errors.epa())
    val_forces_rmse = float(val_fit_errors.forces())

    if rank == 0:
        nit = int(getattr(res, "nit", -1))
        success = int(bool(getattr(res, "success", False)))
        final_loss = float(getattr(res, "fun", float("nan")))

        with open(RESULT_PATH, "a") as f:
            f.write(
                f"{pot_num},"
                f"{train_epa_rmse},{train_forces_rmse},"
                f"{val_epa_rmse},{val_forces_rmse},"
                f"{train_time},{nit},{success},{final_loss}\n"
            )

        print(f"POT: {pot_num}")
        print("SCIPY BFGS TRAIN:", train_fit_errors)
        print("SCIPY BFGS VALID:", val_fit_errors)
        print("SCIPY BFGS TIME:", train_time, "NIT:", nit, "SUCCESS:", bool(success))
        print()
