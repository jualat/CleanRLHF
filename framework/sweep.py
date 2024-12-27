import tyro

import wandb
import yaml
from sac_rlhf import train
from dataclasses import dataclass, replace
from itertools import product


@dataclass
class Args:
    project_name: str = "Hopper-sweep"
    """project name"""
    sweep_count: int = 5
    """sweep count"""
    synthetic_feedback_count: int = 1400
    """number of synthetic feedback"""
    config_filename: str = "sweep_config.yaml"
    """configuration filename"""


def expand_range(param):
    """
    Expands a parameter with a range (min, max) into a list of integers.
    """
    if isinstance(param, dict) and "min" in param and "max" in param:
        return list(range(param["min"], param["max"] + 1))
    elif isinstance(param, dict) and "values" in param:
        return param["values"]
    else:
        raise ValueError(
            "Invalid parameter format. Must include 'min'/'max' or 'values'."
        )


def generate_yaml_valid_config(config_filename="sweep_config.yaml"):
    """
    Generates valid configurations from a YAML file based on parameter constraints.

    Args:
        config_filename (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing filtered, valid parameter configurations.
    """
    try:
        with open(config_filename, "r") as file:
            sweep_config = yaml.safe_load(file)
    except FileNotFoundError:
        raise ValueError(f"Config file '{config_filename}' not found.")
    except yaml.YAMLError:
        raise ValueError("Invalid YAML file format.")

    try:
        values = sweep_config["parameters"]
        total_timesteps = values["total_timesteps"]
        if isinstance(total_timesteps, dict) and "values" in total_timesteps:
            total_timesteps = total_timesteps["values"]  # Extract the integer value
        teacher_feedback_frequency_values = expand_range(
            values["teacher_feedback_frequency"]
        )
        teacher_feedback_num_queries_per_session_values = expand_range(
            values["teacher_feedback_num_queries_per_session"]
        )
        teacher_feedback_batch_size_values = expand_range(
            values["teacher_feedback_batch_size"]
        )

    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}")

    args = tyro.cli(Args)
    valid_configs = []

    for frequency, num_queries, batch_size in product(
        teacher_feedback_frequency_values,
        teacher_feedback_num_queries_per_session_values,
        teacher_feedback_batch_size_values,
    ):
        if num_queries > batch_size:
            synthetic_feedback_count = (total_timesteps // frequency) * num_queries
            if args.synthetic_feedback_count in range(
                synthetic_feedback_count - 100, synthetic_feedback_count + 100
            ):
                valid_configs.append(
                    {
                        "teacher_feedback_frequency": int(frequency),
                        "teacher_feedback_num_queries_per_session": int(num_queries),
                        "teacher_feedback_batch_size": int(batch_size),
                    }
                )

    valid_data = {
        "parameters": {
            "teacher_feedback_frequency": {
                "values": list(
                    set(c["teacher_feedback_frequency"] for c in valid_configs)
                )
            },
            "teacher_feedback_num_queries_per_session": {
                "values": list(
                    set(
                        c["teacher_feedback_num_queries_per_session"]
                        for c in valid_configs
                    )
                )
            },
            "teacher_feedback_batch_size": {
                "values": list(
                    set(c["teacher_feedback_batch_size"] for c in valid_configs)
                )
            },
        }
    }
    return valid_data


def set_valid_yaml_values(filename, valid_values):
    with open(filename, "r") as file:
        yaml_data = yaml.safe_load(file)

    for key, value in valid_values.items():
        if key in yaml_data:
            yaml_data[key] = value
        else:
            yaml_data[key] = value

    with open(filename, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)


def main():
    args = tyro.cli(Args)
    config_filename = args.config_filename
    valid_data = generate_yaml_valid_config(config_filename)
    set_valid_yaml_values(config_filename, valid_data)
    # with open(config_filename, "r") as file:


#     sweep_config = yaml.safe_load(file)
# sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)
# wandb.agent(sweep_id=sweep_id, function=wrapped_train, count=args.sweep_count)


def wrapped_train():
    config = wandb.config

    default_args = Args()

    cmd_args = replace(
        default_args,
        **{field: config[field] for field in config if hasattr(default_args, field)},
    )

    train(cmd_args)


if __name__ == "__main__":
    main()
