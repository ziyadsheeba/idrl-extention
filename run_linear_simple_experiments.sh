#!/bin/sh

SIM_STEPS=10000
NUM_EXPERIMENTS=5

python3 -m src.linear.experiments_active_learning --num-experiments $NUM_EXPERIMENTS --simulation-steps $SIM_STEPS
