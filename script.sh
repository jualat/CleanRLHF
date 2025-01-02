python3 sac_rlhf.py --env-id Ant-v4 \
  --num-envs=1 \
  --capture_video \
  --preference_sampling=uniform \
  --teacher-sim-delta-skip=-1e7 \
  --total_explore_steps=5000 \
  --no-unsupervised_exploration \
  --no-exploration-load \
  --path-to-replay-buffer=models/Ant-v4__sac_rlhf__1__1734788359/10000/replay_buffer.pth \
  --path-to-model=models/Ant-v4__sac_rlhf__1__1734788359/10000/checkpoint.pth