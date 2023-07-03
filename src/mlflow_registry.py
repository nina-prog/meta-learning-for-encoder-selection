"""
Module that handles the mlflow registry
"""

import mlflow
import pandas as pd


def log_model_eval(cv_results, cfg_path, run, verbosity=1):
    """
    Function to log different artefacts and metrics to the active mlflow run

    :param cv_results: pd.DataFrame -- CV results
    :param cfg_path: str -- String to the config file
    :param run: ActiveRun mlflow instance
    :param verbosity: int -- Level of verbosity

    :return: None
    """

    if verbosity > 0:
        print(f"Logging pipeline run to mlflow registry with run name '{run.info.run_name}' ...")

    # Log config file
    mlflow.log_artifact(local_path=cfg_path)

    # Log scores
    for scorer in ["spearman", "r2", "neg_mean_squared_error"]:
        mlflow.log_metric(key=f"cv_test_mean_{scorer}", value=round(cv_results[f'test_{scorer}'].mean(), 4))
        mlflow.log_metric(key=f"cv_test_std_{scorer}", value=round(cv_results[f'test_{scorer}'].std(), 4))


def get_mlflow_tags(X_train: pd.DataFrame, cfg: dict) -> dict:
    """
    Function to log additional tags to the active mlflow run, e.g. size of train data

    :param X_train: pd.DataFrame -- Train Set
    :param cfg: dict -- Parsed config file

    :return: tags: dict -- Key, Value Pairs to add as tags
    """
    tags = {"train_set_size": X_train.shape[0],
            "n_features": X_train.shape[1],
            "k_folds_cv": cfg["modelling"]["k_fold"]}

    return tags
