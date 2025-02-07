#!/bin/bash

poetry run python sac_rlhf.py \
    --tau 0.005 \
    --cuda \
    --q_lr 0.00043582739885087064 \
    --rune \
    --seed 9 \
    --surf \
    --alpha 0.2 \
    --gamma 0.9919472689386292 \
    --track \
    --env_id Swimmer-v5 \
    --autotune \
    --log_file \
    --num_envs 1 \
    --surf_tau 0.999 \
    --log_level INFO \
    --policy_lr 0.0005031039106349765 \
    --rune_beta 174.62233215792418 \
    --batch_size 256 \
    --no-early_stop \
    --lambda_ssl 0.1 \
    --buffer_size 1000000 \
    --wandb_entity "" \
    --no-capture_video \
    --path_to_model "" \
    --learning_starts 5000 \
    --rune_beta_decay 0.00001 \
    --total_timesteps 1000000 \
    --no-exploration_load \
    --policy_frequency 2 \
    --teacher_sim_beta -1 \
    --teacher_sim_gamma 1 \
    --trajectory_length 64 \
    --explore_batch_size 256 \
    --reward_net_dropout 0.2 \
    --wandb_project_name "" \
    --early_stop_patience 5 \
    --early_stopping_mean 0 \
    --early_stopping_step 500000 \
    --evaluation_episodes 10 \
    --preference_sampling disagree \
    --teacher_sim_epsilon 0 \
    --torch_deterministic \
    --total_explore_steps 10000 \
    --evaluation_frequency 10000 \
    --reward_net_val_split 0.2 \
    --path_to_replay_buffer "" \
    --reward_net_hidden_dim 128 \
    --teacher_learning_rate 0.0007688948297561301 \
    --teacher_update_epochs 14 \
    --unlabeled_batch_ratio 4 \
    --surf_sampling_strategy disagree \
    --teacher_batch_strategy full \
    --teacher_minibatch_size 20 \
    --teacher_sim_delta_skip -10000000 \
    --explore_learning_starts 512 \
    --max_augmentation_offset 10 \
    --min_augmentation_offset 5 \
    --teacher_sim_delta_equal 0 \
    --reward_net_hidden_layers 2 \
    --target_network_frequency 1 \
    --unsupervised_exploration \
    --teacher_feedback_schedule linear \
    --actor_and_q_net_hidden_dim 256 \
    --teacher_feedback_batch_size 32 \
    --actor_and_q_net_hidden_layers 4 \
    --teacher_feedback_total_queries 1500 \
    --no-enable_greater_or_smaller_check \
    --teacher_feedback_exponential_lambda 0.1 \
    --teacher_feedback_num_queries_per_session 20
