tensorboard --logdir=runs


python sac_rlhf.py --env_id Hopper-v4 --total_timesteps 10000 --track --wandb_project_name test
python framework/sac_rlhf.py --env_id Hopper-v4 --total_timesteps 10000 --measure_performance pearson
python sac_rlhf.py --env_id Ant-v4 --total_timesteps 10000 --measure_performance pearson
python sac_rlhf.py --env_id Swimmer-v4 --total_timesteps 10000 --measure_performance pearson

python framework/sac_rlhf.py --env_id Hopper-v4 --total_timesteps 10000 --measure_performance pearson
python framework/sac_rlhf.py --env_id Hopper-v4 --total_timesteps 30000 --measure_performance pearson
python sac_rlhf.py --env_id Hopper-v4 --total_timesteps 50000 --measure_performance pearson
python sac_rlhf.py --env_id Hopper-v4 --total_timesteps 50000 --measure_performance pearson --capture_video


python sweep.py --project_name "Ant-v4" --config_filename "ant_sweep.yaml"



python sac_rlhf.py --env_id Hopper-v4 --total_timesteps 50000 --measure_performance pearson --teacher_feedback_num_queries_per_session 10 --teacher_feedback_mode human



framework/videos/Hopper-v4__sac_rlhf__1__1736861266/trajectories/trajectory_887_919_0-episode-0.mp4                                         