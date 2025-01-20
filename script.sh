poetry run python sac_rlhf.py --env-id=Hopper-v5 \
  --num-envs=1 \
  --exp_name=Junis-Test \
  --wandb_project_name=Junis-Test \
  --no-track \
  #--rune \
  --no-capture_video \
  --log_level=INFO \
  --total_timesteps=1000000 \
  --preference_sampling=disagree \
  --teacher-sim-delta-skip=-1e7 \
  --total_explore_steps=10000 \
  --unsupervised_exploration \
  --no-exploration-load \
  --path-to-replay-buffer=models/Ant-v5__Benjamin-Test__1__1737145974/10000/replay_buffer.pth \
  --path-to-model=models/Ant-v5__Benjamin-Test__1__1737145974/10000/checkpoint.pth