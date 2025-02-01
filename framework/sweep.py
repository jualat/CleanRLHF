from dataclasses import dataclass, fields, replace

import tyro
import wandb
import yaml
from ppo_original import Args as ppoArgs
from ppo_original import train as run_ppo
from prefppo import Args as prefppoArgs
from prefppo import run as run_prefppo
from sac_original import Args as sacArgs
from sac_original import train as run_sac
from sac_rlhf import Args as sac_rlhfArgs
from sac_rlhf import run as run_sac_rlhf


@dataclass
class BaseArgs:
    project_name: str = ""
    """project name"""
    sweep_count: int = 3
    """sweep count"""
    config_filename: str = ""
    """configuration filename"""
    entity: str = "cleanRLHF"
    """wandb entity name"""
    sweep_id: str = ""
    """sweep id"""
    algorithm: str = "sac-rlhf"
    """algorithm to run (sac-rlhf, pref-ppo, sac, ppo)"""


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
    run = wandb.init()
    args = tyro.cli(BaseArgs)
    try:
        config = wandb.config

        if args.algorithm == "pref-ppo":
            cmd_args = replace(
                prefppoArgs(),
                **{
                    field.name: config[field.name]
                    for field in fields(prefppoArgs)
                    if field.name in config
                },
            )
        elif args.algorithm == "sac-rlhf":
            cmd_args = replace(
                sac_rlhfArgs(),
                **{
                    field.name: config[field.name]
                    for field in fields(sac_rlhfArgs)
                    if field.name in config
                },
            )
        elif args.algorithm == "sac":
            cmd_args = replace(
                sacArgs(),
                **{
                    field.name: config[field.name]
                    for field in fields(sacArgs)
                    if field.name in config
                },
            )
        elif args.algorithm == "ppo":
            cmd_args = replace(
                ppoArgs(),
                **{
                    field.name: config[field.name]
                    for field in fields(ppoArgs)
                    if field.name in config
                },
            )

        if args.algorithm == "pref-ppo":
            run_prefppo(cmd_args)
        elif args.algorithm == "sac-rlhf":
            run_sac_rlhf(cmd_args)
        elif args.algorithm == "sac":
            run_sac(cmd_args)
        elif args.algorithm == "ppo":
            run_ppo(cmd_args)
        else:
            raise ValueError(f"Algorithm {args.algorithm} not supported")
    except Exception as e:
        wandb.log({"error": str(e)})
        raise e
    finally:
        run.finish()


if __name__ == "__main__":
    main()
