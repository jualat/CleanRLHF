#!/bin/bash
#SBATCH --job-name=CleanRLHFSweep
#SBATCH --comment="Running WandB sweep for RLHF course"
#SBATCH --partition=NvidiaAll
#SBATCH --mail-user=b.kuen@campus.lmu.de
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/k/kuen/workspace/cleanrlhf/framework
#SBATCH --output=/home/k/kuen/workspace/cleanrlhf/framework/slurm/slurm.%j.%N.out
#SBATCH --error=/home/k/kuen/workspace/cleanrlhf/framework/slurm/slurm.%j.%N.err
#SBATCH --ntasks=1
date;hostname;id;pwd

entity='cleanRLHF'
echo 'entity:' $entity

project_name='Slurm_test'
echo 'project_name:' $project_name

sweep_config='./sweep_config/slurm_sweep.yaml'
echo 'sweep_config:' $sweep_config

echo 'activating virtual environment'
source /home/k/kuen/miniconda3/bin/activate
conda activate py310
which python3.10

python3.10 sweep.py --project $project_name --entity $entity -sweep_count 3 --config_filename $sweep_config