#!/bin/bash
#SBATCH --job-name CleanRLHFSweep
#SBATCH --comment "Running WandB sweep for RLHF course"
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --mail-user <b.kuen@campus.lmu.de>
#SBATCH --mail-type=BEGIN
date;hostname;id;pwd

entity='cleanRLHF'
echo 'entity:' $entity

project_name='Slurm_test'
echo 'project_name:' $project_name

sweep_config='../sweep_config/slurm_sweep.yaml'
echo 'sweep_config:' $sweep_config

echo 'activating virtual environment'
source ~/miniconda3/bin/activate
conda activate py310
which python3.10

python3.10 sweep.py --project $project_name --entity $entity -sweep_count 3 --config_filename $sweep_config