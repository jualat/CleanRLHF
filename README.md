# CleanRLHF

A framework that implements reinforcement learning from human feedback following the [PEBBLE paper](https://arxiv.org/abs/2106.05091). 

## Installation

Clone this repository and cd into it.

```
git clone https://github.com/jualat/CleanRLHF.git
cd CleanRLHF
```

Make sure to have [poetry](https://python-poetry.org/docs/#installation) installed on your machine. Install all requirements via
```
poetry install
```

For contributing, install [pre-commit](https://pre-commit.com/#install) on your machine and install the hooks in your local `.git` directory using

```
pre-commit install --allow-missing-config
```

Note: the [flag](https://github.com/pre-commit/pre-commit/issues/984) will allow you to push to branches other than main, even if the code doesn't fulfill all pre-commit requirements (i.e. the existence of a `.pre-commit-config.yaml` file is not mandatory in that branch).

## Usage

To run the implementation, execute
```
python3 framework/sac_rlhf.py
```

There exist several arguments which can be found in `framwork/sac_rlhf.py`. For example, to set the [gymnasium environment](https://gymnasium.farama.org/index.html), use the `--env_id` argument as in
```
python3 framework/sac_rlhf.py --env_id Walker2d
```

# Tensorboard
By running `tensorboard --logdir=runs` a local instance of TensorBoard will be started (http://localhost:6006/)

## Performance metrics
### Pearson correlation
By using the flag `--measure_performance pearson`, an additional graph displaying the Pearson correlation will be generated and made available in TensorBoard for visualization.