# CleanRLHF

## Prerequisites

Ensure the following are installed on your system:
- Python 3.10+
- xvfb (`sudo apt install xvfb` on Debian-based systems)
- ffmpeg (`sudo apt install ffmpeg` on Debian-based systems)

## Running the code

To run the code, execute the following command, replacing `Hopper-v4` with the desired environment ID:

```bash
python3 sac_rlhf.py --env-id Hopper-v4
```

Use the `--capture-video` flag to record a video of the agent's performance as well as rendering 
the selected trajectories for human preference:

```bash
python3 sac_rlhf.py --env-id Hopper-v4 --capture-video
```

Use the `xvfb-run` command to execute the script in a headless Linux environment:

```bash
xvfb-run -- python3 sac_rlhf.py --env-id Hopper-v4 --capture-video
```

For a full list of available command-line arguments, take a look at the `sac_rlhf.py` script.

### Tracking

Use the `--track  --wandb-project-name HopperTest --wandb-entity cleanRLHF` flag to activate tracking via Weights &
Biases

```bash
wandb login
python3 sac_rlhf.py --track  --wandb-project-name HopperTest --wandb-entity cleanRLHF
```