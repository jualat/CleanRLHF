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

echo 'activating virtual environment'
source /home/k/kuen/miniconda3/bin/activate
conda activate py310
which python3.10

echo 'start sweep script'
python3.10 sweep.py --project_name Slurm_test --entity cleanRLHF --sweep_count 3 --config_filename ./sweep_config/slurm_sweep.yaml