#!/bin/bash -l

module load cmake/3.21.3
module load openmpi/4.1.4
module load gnu9/9.3
module load prun/1.2
module load Python

conda activate peter_env

prun /home/peivzarenkov/.conda/envs/peter_env/bin/python train_mtp_my_adam.py
# prun /home/peivzarenkov/.conda/envs/peter_env/bin/python test.py
