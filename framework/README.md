# CleanRLHF

A framework that implements reinforcement learning from human feedback. 

##### Table of Contents
<!--TODO-->

## Introduction 👋

This project is focussed on reimplementing [PEBBLE](https://arxiv.org/abs/2106.05091), a framework for RLHF. It is based on [SAC](https://arxiv.org/abs/1801.01290), an off-policy actor-critic algorithm for deep RL.

## Performance 🚀

## Getting started 💡

### Prerequisites

Ensure the following are installed on your system:

* [poetry](https://python-poetry.org/docs/#installation)
* Python 3.10+
* xvfb (`sudo apt install xvfb` on Debian-based systems)
* ffmpeg (`sudo apt install ffmpeg` on Debian-based systems)

If you want to contribute, you will also need
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

## Usage

### Basic Code Execution

To run the code, execute the following command from the `framework` directory, replacing Hopper-v4 with the desired [environment ID](https://gymnasium.farama.org/environments/mujoco/):

```sh
python3 sac_rlhf.py --env-id Hopper-v4
```

Use the xvfb-run command to execute the script in a headless Linux environment:

```sh
xvfb-run -- python3 sac_rlhf.py --env-id Hopper-v4 --capture-video
```

### Features

For a full list of available command-line arguments, take a look at `sac_rlhf.py` or run:
```sh
python3 sac_rlhf.py -h
```

#### SURF

#### RUNE

#### Tracking

#### Video Recording

#### Model Saving/Loading

#### Tensorboard

#### Hyperparameter Tuning
