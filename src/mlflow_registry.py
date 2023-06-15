"""
Module that handles the mlflow registry
"""

import mlflow


def log_model_eval(cv_results, cfg, cfg_path, verbosity=1):
    """
    Function to log different artefacts and metrics to the active mlflow run

    :param cv_results: pd.DataFrame -- cv_results
    :param cfg: dict -- Parsed config file
    :param cfg_path: str -- String to the config file

    :return: None
    """

    if verbosity > 0: print("Logging model results and specifications to mlflow registry ...")

    # Log config file
    mlflow.log_artifact(local_path=cfg_path)

    # Log scores
    for scorer in cfg["modelling"]["scoring"]:
        mlflow.log_metric(key=f"cv_test_mean_{scorer}", value=round(cv_results[f'test_{scorer}'].mean(), 4))
        mlflow.log_metric(key=f"cv_test_std_{scorer}", value=round(cv_results[f'test_{scorer}'].std(), 4))
