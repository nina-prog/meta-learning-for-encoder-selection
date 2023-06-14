"""
Module for the pipeline. Execute via python3 main.py --config configs/config.yaml
"""
import time

import mlflow.sklearn

import src.utils
import src.load_datasets
import src.modelling
import src.mlflow_registry


def main():
    """
    Function that executes the pipeline
    :return:
    """

    # Parse config file
    # Track time for execution
    start_time = time.time()
    print(20 * "=" + " Starting pipeline " + 20 * "=")

    # Read config file
    cfg, cfg_path = src.utils.parse_args()
    verbosity = cfg["general"]["verbosity"]

    # Load Data
    dataset = src.load_datasets.load_dataset(path=cfg["paths"]["dataset_path"], verbosity=verbosity)
    rankings = src.load_datasets.load_rankings(path=cfg["paths"]["rankings_path"], verbosity=verbosity)

    """
    Add here pipeline steps, e.g. preprocessing, fitting, predictions ...
    """

    # Split data into train test
    X_train, X_test, y_train, y_test = src.modelling.train_test_split_data(train_data=dataset,
                                                                           split_size=cfg["modelling"]["split_size"])

    # Log model evaluation to mlflow registry
    mlflow.sklearn.autolog(log_models=False)
    with mlflow.start_run(tags = {"train_set_size": len(X_train)}):
        # Perform CV and train model
        model, cv_result = src.modelling.train_model(model=cfg["modelling"]["model"],
                                                     train_data=X_train,
                                                     train_labels=y_train,
                                                     scoring=cfg["modelling"]["scoring"],
                                                     hyperparam_grid=None,
                                                     verbosity=verbosity,
                                                     k_fold=cfg["modelling"]["k_fold"])
        src.mlflow_registry.log_model_eval(cv_result, cfg, cfg_path)

    # Make predictions
    src.modelling.make_prediction(model=model, test_data=X_test,
                                  result_path=cfg["paths"]["result_path"], verbosity=verbosity)

    # Track time for total runtime and display end of pipeline
    runtime = time.time() - start_time
    print(20 * "=" + f" Pipeline finished ({src.utils.display_runtime(runtime)}) " + 20 * "=")


if __name__ == "__main__":
    main()
