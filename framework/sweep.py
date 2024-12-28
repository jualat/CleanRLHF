import tyro

import wandb
import yaml
from sac_rlhf import train, Args
from dataclasses import dataclass, fields, replace


@dataclass
class BaseArgs:
    project_name: str = "Hopper-tuning"
    """project name"""
    sweep_count: int = 3
    """sweep count"""
    config_filename: str = "sweep_config.yaml"
    """configuration filename"""
    entity: str = "cleanRLHF"
    """wandb entity name"""
    sweep_id: str = ""
    """sweep id"""


def main():
    args = tyro.cli(BaseArgs)
    config_filename = args.config_filename
    if args.sweep_id:
        wandb.agent(
            sweep_id=args.sweep_id,
            function=wrapped_train,
            count=args.sweep_count,
            project=args.project_name,
            entity=args.entity,
        )
    else:
        with open(config_filename, "r") as file:
            sweep_config = yaml.safe_load(file)
        sweep_id = wandb.sweep(
            sweep=sweep_config, project=args.project_name, entity=args.entity
        )
        wandb.agent(sweep_id=sweep_id, function=wrapped_train, count=args.sweep_count)


def wrapped_train():
    # Initialize the W&B run
    run = wandb.init()

    try:
        # Access sweep configuration
        config = wandb.config

        # Create Args object with sweep parameters
        cmd_args = replace(
            Args(),
            **{
                field.name: config[field.name]
                for field in fields(Args)
                if field.name in config
            }
        )

        # Pass Args to the train function in sac_rlhf.py
        train(cmd_args)
    except Exception as e:
        # Log errors to W&B
        wandb.log({"error": str(e)})
        raise e
    finally:
        # Finish the W&B run
        run.finish()


if __name__ == "__main__":
    main()
