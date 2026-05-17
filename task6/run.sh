#!/bin/bash -l

module load cmake/3.21.3
module load openmpi/4.1.4
module load gnu9/9.3
module load prun/1.2
module load Python

conda activate peter_env

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"

prun "$PYTHON_BIN" train_etn_my_sgd.py
prun "$PYTHON_BIN" train_etn_my_momentum.py
prun "$PYTHON_BIN" train_etn_my_adam.py
prun "$PYTHON_BIN" train_etn_my_muon_pad_sqrt.py
prun "$PYTHON_BIN" train_etn_my_muon_factorization.py
prun "$PYTHON_BIN" train_etn_scipy_bfgs.py
