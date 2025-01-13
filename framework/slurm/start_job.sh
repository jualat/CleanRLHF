cd ~/workspace/cleanrlhf/framework
read -s WANDB_API_KEY && sbatch --export=WANDB_API_KEY=$WANDB_API_KEY wandb_sweep.sbatch