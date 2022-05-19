#!/bin/sh

SIM_STEPS=1000
NUM_EXPERIMENTS=5

python3 -m src.linear.simulate_policy_learning --num-experiments $NUM_EXPERIMENTS --simulation-steps $SIM_STEPS
