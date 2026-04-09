#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from mpi4py import MPI
import sys

sys.path.insert(0, "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP")

from mlip_4 import LossFunction, RadialBasisCinf, MTP
import utils

TRAIN_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/PdAg.json"
VALID_PATH = "/home/peivzarenkov/mlip-4/Testing-optimization-methods-in-MLP/datasets/PdAg/validation_PdAg.json"
RESULT_PATH = "results/results_mtp_my_sgd_clip_lr_grid_experiment.csv"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

train_json_bytes = b""
valid_json_bytes = b""

if rank == 0:
    with open(RESULT_PATH, "w") as f:
        f.write(
            "clip,lr,pot_num,"
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
max_steps = 1000
gtol = 1e-6
full_batch = False

clips = [1, 3, 5, 10, 15]
lrs = [1e-3, 3e-3, 1e-2]

for clip in clips:
    for lr in lrs:
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
            opt = utils.optimizers.SGD(lr_schedule=lr_schedule)

            trainer = utils.mlip_trainer.MlipTrainer(gtol=gtol, max_steps=max_steps)

            comm.Barrier()
            start_timer = time.perf_counter()

            hist = trainer.train(
                pot=pot,
                train_func=train_func,
                optimizer=opt,
                batcher=batcher,
                full_batch=full_batch,
                clip=float(clip),
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

            if rank == 0:
                with open(RESULT_PATH, "a") as f:
                    f.write(
                        f"{clip},{lr},{pot_num},"
                        f"{train_epa_rmse},{train_forces_rmse},"
                        f"{val_epa_rmse},{val_forces_rmse},"
                        f"{train_time},{steps_done},{epochs_done},{final_loss}\n"
                    )

                print(f"CLIP: {clip} LR: {lr} POT: {pot_num}")
                print("MY SGD TRAIN:", train_fit_errors)
                print("MY SGD VALID:", val_fit_errors)
                print("MY SGD TIME:", train_time, "STEPS:", steps_done, "EPOCHS:", epochs_done, "FINAL_LOSS:", final_loss)
                print()
