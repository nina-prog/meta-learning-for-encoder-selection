""" This module contains utility functions / helper functions """
import argparse
import yaml
import os


def load_config(config_path):
    """
    Loads a YAML configuration file.

    :param config_path: Path to the configuration file
    :type config_path: str

    :return: Configuration dictionary
    :rtype: dict
    """
    try:
        with open(config_path, "r") as ymlfile:
            return yaml.load(ymlfile, yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {config_path} not found!")
    except PermissionError:
        raise PermissionError(f"Insufficient permission to read {config_path}!")
    except IsADirectoryError:
        raise IsADirectoryError(f"{config_path} is a directory!")


def parse_args():
    """
    Function that parses the config.yaml and return the cfg as dict
    :return: dict -- parsed config file, str -- path to the config file
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)
    cfg_path = args.config
    return cfg, cfg_path


def check_file_exists(file_path):
    """
    Checks if a file exists.

    :param file_path: Path to the file
    :type file_path: str

    :return: None

    :raises FileNotFoundError: If the file does not exist
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} is missing.")


def display_runtime(runtime) -> str:
    """
    Helper function to display the runtime of our pipeline, i.e. if runtime > 60 seconds output minutes.
    :param runtime: runtime in seconds
    :return: String to display the runtime in seconds or minutes
    """
    if runtime > 60:
        runtime = runtime / 60
        return f"took {runtime:.2f} minutes"
    else:
        return f"took {runtime:.2f} seconds"
