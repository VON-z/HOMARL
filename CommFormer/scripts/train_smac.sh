#!/bin/sh
env="StarCraft2"
algo="commformer_dec"
exp="single"
seed=1
name="CommFormer"

map="3s5z"
ppo_epochs=15
ppo_clip=0.05
steps=200000

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo}  --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 \
    --episode_length 100 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_inteval 10000 \
    --use_value_active_masks --use_eval --prefix_name ${name} --use_bilevel --use_wandb
