import yaml
import torch
import torch.utils.data


def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    load_config(yaml_file="./config.yml")
