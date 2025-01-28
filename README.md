# CleanRLHF

This project is focussed on implementing a framework for Reinforcement Learning from Human Feedback.

* [Introduction](#-introduction)
* [Performance](#-performance)
* [Getting Started](#-getting-started)
  * [Prerequisites](#prerequisites)
  * [Installing the Dependencies](#installing-the-dependencies)
* [Usage](#-usage)
  * [Basic Code Execution](#basic-code-execution)
  * [Human Feedback](#human-feedback)
* [Features](#-features)
    * [Unsupervised Exploration](#unsupervised-exploration)
    * [Trajectory Sampling](#trajectory-sampling)
    * [Trajectory Scheduling](#trajectory-scheduling)
    * [SURF](#surf)
    * [RUNE](#rune)
    * [Video Recording](#video-recording)
    * [Model Saving/Loading](#model-savingloading)
    * [Tracking](#tracking)
    * [Hyperparameter Tuning](#hyperparameter-tuning)
    * [Toolbox](#toolbox)


## 👋 Introduction

This framework implements RLHF and is oriented towards [PEBBLE](https://arxiv.org/abs/2106.05091). It is based on [SAC](https://arxiv.org/abs/1801.01290), an off-policy actor-critic algorithm for deep RL.

There also exists an implementation of [PrefPPO](https://arxiv.org/abs/1706.03741) (WIP), which is a faster but less efficient approach to RLHF. Additionally to the implementations of these two papers, several features have been added, which are aimed at improving the performance as well as the user experience of our program. The features are [discuessed](#-performance) and are [documented](#-features) below.


## 🚀 Performance
(WIP)
<!--TODO-->


## 💡 Getting Started

### Prerequisites

Ensure the following are installed on your system:

* [poetry](https://python-poetry.org/docs/#installation)
* Python 3.10+
* xvfb (`sudo apt install xvfb` on Debian-based systems)
* ffmpeg (`sudo apt install ffmpeg` on Debian-based systems)

If you want to contribute, then you'll also need:
* [pre-commit](https://pre-commit.com/#install)

### Installing the Dependencies

Clone this repository and cd into it if you haven't done so already:

```sh
git clone https://github.com/jualat/CleanRLHF.git
cd CleanRLHF
```

Execute this command to install all dependencies automatically:

```sh
poetry install
```


## 🎯 Usage

### Basic Code Execution

To run the code, execute the following command from the `framework` directory, replacing Hopper-v5 with the desired environment ID (from either [Mujoco](https://gymnasium.farama.org/environments/mujoco/) or [Deepmind Control Suite](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/README.md)):

```sh
python3 sac_rlhf.py --env-id Hopper-v5
```

Use the xvfb-run command to execute the script in a headless Linux environment:

```sh
xvfb-run -- python3 sac_rlhf.py --env-id Hopper-v5
```

Setting the right hyperparameters is crucial for performance. Consider looking at our methods for [hyperparameter tuning](#hyperparameter-tuning) and/or check whether there exists a script `../[ENV_ID].sh` with the hyperparameters that we found best for a specific environment.

### Human Feedback

## ✈️ Features

For a full list of available command-line arguments, take a look at `sac_rlhf.py` or run:
```sh
python3 sac_rlhf.py -h
```

### Unsupervised Exploration

At the beginning of RLHF training, the models are initialized randomly and thus don't always explore states sufficiently. Since informative queries in the early phase of training are essential for its success, this is a major challenge. [Lee et al.](https://arxiv.org/abs/2106.05091) handle this issue using the environment's state entropy as an intrinsic reward in unsupervised pre-exploration that can be carried out before the actual RLHF training begins.

Unsupervised pre-training is enabled by default and can be disabled with the `--no-unsupervised-exploration` flag.

It suffices to do pre-exploration only once per environment on your machine; see [Model Saving/Loading](#model-savingloading).

### Trajectory Sampling

Every trajectory that the agent produces during the training is saved in the replay buffer. Sampling random pairs of such trajectories when consulting the human expert is inefficient. [Lee et al.](https://arxiv.org/abs/2106.05091) propose two methods for more profitable trajectory sampling. Note that the methods utilize ensemble reward models, a technique which we employ for increased training stability:

* **Disagree**: We use an ensemble of three reward models to stabilize the training. If, given a pair of trajectories, the models are not sure which one to prefer (i.e. the standard deviation of the preferences given by the members of the ensemble is high), then it is efficient to ask the human expert.
* **Entropy**: If a pair of trajectories is near the decision boundary, that is, the entropy of the preferences given by the members of the ensemble is high, then we prefer to consult the human teacher for this pair.

By default, disagreement sampling is used. It can be set using the `--preference-sampling` flag; values must be 'uniform', 'disagree', or 'entropy'.

### Trajectory Scheduling

Because of the low knowledge of the models at the beginning of the training, it can make sense to consult the human expert more often in earlier phases of the training.

Scheduling must be either 'linear' or 'exponential' and is set to be the latter by default. It can be adjusted with the `--teacher-feedback-schedule` flag.

### SURF

Collecting human feedback on a scale is pretty costly. An idea proposed by [Park et al.](https://arxiv.org/abs/2203.10050) is to use data augmentation to extract the highest possible amount of information from human labels.

SURF is enabled by default and can be disabled with the `--no-surf` flag.

### RUNE

The exploration/exploitation trade-off is a problem central to RL. [Liang et al.](https://arxiv.org/abs/2205.12401) present an intrinsic reward to encourage exploration.

RUNE is disabled by default and can be enabled with the `--rune` flag.

### Video Recording

Use the `--capture-video` flag to record a video of the agent's performance as well as rendering 
the selected trajectories for human preference:

```sh
python3 sac_rlhf.py --env-id Hopper-v5 --capture-video
```

### Model Saving/Loading

Once you have done unsupervised pre-exploration, the replay buffer and the model are automatically saved in `./models/[RUN]/[EXPLORATION_STEPS]/`.
To save time, instead of exploring the same environment every time, you can now load both results, e.g.:

```sh
python3 sac_rlhf.py --exploration-load --path-to-replay-buffer=models/myrun/10000/replay_buffer.pth --path-to-model=models/myrun/10000/checkpoint.pth
```

Note that the states of both objects are also saved at the end of a run or on KeyBoardInterrupt.

### Tracking

#### [Weights & Biases](https://wandb.ai/)

Use the `--track  --wandb-project-name HopperTest --wandb-entity cleanRLHF` flag to activate tracking via Weights &
Biases:

```bash
wandb login
python3 sac_rlhf.py --track  --wandb-project-name HopperTest --wandb-entity cleanRLHF
```

#### [Tensorboard](https://www.tensorflow.org/tensorboard)

Run the following command to start a local instance of TensorBoard:

```sh
tensorboard --logdir=runs
```

You can access the TensorBoard at http://localhost:6006/

### Hyperparameter Tuning

The `sweep.py` script automates hyperparameter optimization using Weights & Biases (W&B) Sweeps. 

#### How to Run a Sweep:
1. Specify the hyperparameter config:

   Place your sweep configuration in `sweep_config/<SWEEP_NAME>.yaml`.


2. Run the sweep with W&B:

   ```bash
   python3 sweep.py --project-name <PROJECT_NAME> --entity <WAND_ENTITY> --sweep-count 3 --config-filename ./sweep_config/<SWEEP_NAME>.yaml
   ```
   `--sweep-count`:  Number of runs to launch in this session.
   `--project-name`: The name of your W&B project.
    `--entity`: Your W&B entity (team or username).


3. Run the Sweep with Sweep ID

    ```bash
   python3 sweep.py --project-name --sweep_id <SWEEP_ID> --sweep_count 3 
   ```

    `--sweep_id`:  You can find this ID on the W&B dashboard.

#### How to Run a Sweep on SLURM:

1. Setup a virtual environment and install the dependencies:

   > Note: This step is only required once. It'll install Miniconda, installs Python3.10. creates a virtual environment
   > and executes the `poetry install` command.
   > It also clones the CleanRLHF repository into ~/workspace directory

   ```bash
   cd slurm
   sh setup_venv.sh
   ```

2. Adjust `slurm/wandb_sweep.sbatch` to your needs. Make sure to replace the email to receive notifications and update 
   absolute paths.

   ```
   #SBATCH --mail-user=b.kuen@campus.lmu.de
   #SBATCH --mail-type=ALL
   #SBATCH --chdir=/home/k/kuen/workspace/cleanrlhf/framework
   #SBATCH --output=/home/k/kuen/workspace/cleanrlhf/framework/slurm/slurm.%j.%N.out
   #SBATCH --error=/home/k/kuen/workspace/cleanrlhf/framework/slurm/slurm.%j.%N.err
   ```
   
   Replace `python3.10 sweep.py --project_name Ant_common_tuning --entity cleanRLHF --sweep_count 3 --config_filename ./sweep_config/ant_sweep.yaml`
   with the startup command of your sweep.

3. Submit the job to the SLURM cluster:

   > Note: You'll be prompted to paste your WAND API key. You can find it on your W&B dashboard.

   ```bash
   sh start_job.sh
   ```
   
   The command returns a job ID. You can check the status of your job with `squeue -u <username>`.

### Toolbox

In addition to the framework, we have created a toolbox to compare runs of the framework. The tool as well as its documentation can be found in the [`toolbox`](https://github.com/jualat/CleanRLHF/tree/main/toolbox#readme) subdirectory.


