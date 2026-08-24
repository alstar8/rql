#!/bin/bash
RL=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
source /home/jovyan/users/staroverov/env.sh
cd /home/jovyan/users/staroverov/GuideVLA
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -p /home/jovyan/users/staroverov/envs/archer     python $RL/scripts/probe_divergence.py --repo /home/jovyan/users/staroverov/GuideVLA     --seed $s --fix-gripper --out $RL/runs/probe/worker_fix_s$s.json
done
touch $RL/runs/probe/worker_fix.DONE2
