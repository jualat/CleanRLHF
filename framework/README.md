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

### Hyperparameter Tuning

The sweep.py script automates hyperparameter optimization using Weights & Biases (W&B) Sweeps. 

#### How to Run a Sweep:
1. **Specify the hyperparameter config:**

   Place your sweep configuration in `sweep_config/<SWEEP_NAME>.yaml`.


2. **Run the sweep with W&B:**

   ```bash
   python sweep.py --project_name <PROJECT_NAME> --entity <WAND_ENTITY> --sweep_count 3 --config_filename ./sweep_config/<SWEEP_NAME>.yaml
   ```
   `--sweep_count`:  Number of runs to launch in this session.
   `--project_name`: The name of your W&B project.
    `--entity`: Your W&B entity (team or username).


3. **Run the Sweep with Sweep ID**

    ```bash
   python sweep.py --sweep_id <SWEEP_ID> --sweep_count 3 
   ```
    `--sweep_id`:  You can find this ID on the W&B dashboard.