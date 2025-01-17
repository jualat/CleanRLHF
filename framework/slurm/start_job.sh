cd ~/workspace/cleanrlhf/framework
echo 'Enter your WANDB Token' && read -s WANDB_API_KEY
sbatch --export=WANDB_API_KEY=$WANDB_API_KEY,MUJOCO_GL=egl wandb_sweep.sbatch