"""
Module for the pipeline. Execute via python3 main.py --config configs/config.yaml
"""
import time

import src.utils
import src.load_datasets


def main():
    """
    Function that executes the pipeline
    :return:
    """

    # Parse config file
    # Track time for execution
    start_time = time.time()
    print(20*"=" + " Starting Pipeline " + 20*"=")

    # Read config file
    cfg = src.utils.parse_args()
    # Load Data
    dataset = src.load_datasets.load_dataset(path = cfg["paths"]["dataset_path"])
    rankings = src.load_datasets.load_rankings(path = cfg["paths"]["rankings_path"])

    """
    Add here pipeline steps, e.g. preprocessing, fitting, predictions ...
    
    # Preprocessing
    # Fitting
    # ...
    
    """

    # Track time for total runtime and display end of pipeline
    runtime = time.time() - start_time
    print(20 * "=" + f" Pipeline Finished ({src.utils.display_runtime(runtime)}) " + 20 * "=")


if __name__== "__main__":
    main()