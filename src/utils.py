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
    Function that parses the arguments
    :return: Parsed arguments
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file")
    parser.add_argument("--subsample", default=None, type=int, help="Number of samples to use")
    args = parser.parse_args()

    return args


def parse_config(args):
    """
    Parses the config file, given the parsed arguments
    :param args: Parsed arguments
    :return: dict: parsed config file, str: path to the config file
    """
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
    :return: String to display the runtime in seconds and/or minutes
    """
    if runtime > 60:
        minutes = runtime // 60
        seconds = runtime % 60
        return f"took {int(minutes)}min and {int(seconds)}sec"
    else:
        return f"took {runtime:.2f} seconds"

