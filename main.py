"""
Module for the pipeline.
Execute via python3 main.py --config configs/config.yaml
or if you want to run pipeline with subsample of size 100
python3 main.py --config configs/config.yaml --subsample 100
"""
import time
import pandas as pd
import numpy as np

import src.utils
import src.load_datasets

import src.encoding
import src.evaluate_regression
import src.pointwise_method
import src.pairwise_method
import src.neural_net

from src.data_cleaning import drop_pearson_correlated_features
from src.meta_information import add_dataset_meta_information


def main():
    """
    Function that executes the pipeline
    """

    # Track time for execution
    start_time = time.time()
    print(30 * "=" + " PIPELINE STARTED " + 30 * "=")

    # Parse arguments and read config file cfg
    args = src.utils.parse_args()
    cfg, cfg_path = src.utils.parse_config(args)
    verbosity = cfg["general"]["verbosity"]

    ### LOAD DATA ###
    df_train = src.load_datasets.load_dataset(path=cfg["paths"]["train_rank_data_path"],
                                              verbosity=verbosity,
                                              subsample=args.subsample)

    # Load Test data (Hold-out-set) that is used for course-internal scoring
    X_test = src.load_datasets.load_test_data(path=cfg["paths"]["test_values_path"],
                                              verbosity=verbosity,
                                              subsample=args.subsample)

    # Check method to use
    # Valid arguments are 'regression', 'pointwise', 'pairwise', 'listwise'
    # For each method preprocess data, train a model and predict results
    if verbosity > 0:
        print(f"Running Pipeline for prediction type: {cfg['general']['method']}")

    if cfg["general"]["method"] == "regression":
        # ToDo
        pass

    elif cfg["general"]["method"] == "pointwise":
        # Constants for supervisors functions
        FACTORS = ["dataset", "model", "tuning", "scoring"]
        NEW_INDEX = "encoder"
        TARGET = "rank"

        # Preprocess data
        X_train, y_train, X_test = src.pointwise_method.preprocessing(df_train=df_train,
                                                                      X_test=X_test,
                                                                      config=cfg)

        # Get indices for custom cross validation that is used within the modelling.py module
        indices = src.evaluate_regression.custom_cross_validated_indices(df_train,
                                                                         FACTORS,
                                                                         TARGET,
                                                                         n_splits=cfg["modelling"]["k_fold"],
                                                                         shuffle=True,
                                                                         random_state=1444)

        # Train model and predict results of test data
        model = src.pointwise_method.modelling(X_train=X_train,
                                               y_train=y_train,
                                               indices=indices,
                                               config_path=cfg_path,
                                               config=cfg)
        src.pointwise_method.prediction_pointwise(model=model,
                                                  X_test=X_test,
                                                  config=cfg)
    elif cfg["general"]["method"] == "pairwise":
        # Constants for supervisors functions
        FACTORS = ["dataset", "model", "tuning", "scoring"]
        NEW_INDEX = "encoder"
        TARGET = "rank"

        # Preprocess data
        X_train, y_train, X_test, base_df = src.pairwise_method.preprocessing(df_train=df_train,
                                                                              X_test=X_test,
                                                                              config=cfg)

        # Get indices for custom cross validation that is used within the modelling.py module
        indices = src.evaluate_regression.custom_cross_validated_indices(pd.concat([X_train, y_train], axis=1),
                                                                         list(X_train.columns),  # factors
                                                                         list(y_train.columns),  # target
                                                                         n_splits=cfg["modelling"]["k_fold"],
                                                                         shuffle=True,
                                                                         random_state=1444)

        # Train model and predict results of test data
        model = src.pairwise_method.modelling(X_train=X_train,
                                              y_train=y_train,
                                              base_df=base_df,
                                              indices=indices,
                                              config_path=cfg_path,
                                              config=cfg)

        merge_cols = list(base_df)
        merge_cols.remove("rank")
        merge_cols.remove("encoder")
        src.pairwise_method.prediction_pointwise(model=model,
                                                 X_test=X_test,
                                                 target_columns=list(y_train.columns),
                                                 merge_cols=merge_cols,
                                                 config=cfg)

    elif cfg["general"]["method"] == "listwise":
        # Constants for supervisors functions
        FACTORS = ["dataset", "model", "tuning", "scoring"]
        NEW_INDEX = "encoder"

        if "cv_score" in df_train.columns:
            df_train = df_train.drop("cv_score", axis=1)

        # Bring train data in desired shape for listwise pred
        df_train = pd.pivot(df_train, index=FACTORS, columns="encoder", values="rank").reset_index()
        # TODO: SAME FOR X_TEST BUT HERE SEEMS TO BE AN ERROR? SUPERVISOR MISSING COLUMN 'ENCODER' ???
        X_test = pd.pivot(X_test, index=FACTORS, columns="encoder").reset_index()
        X_test.columns = X_test.columns.droplevel(level="encoder")
        X_test = X_test[FACTORS]

        # Get train data
        X_train = df_train[FACTORS]
        y_train = df_train.drop(FACTORS, axis=1)
        y_train = y_train.fillna(np.max(y_train))

        # Save unprocessed df for building average spearman score
        X_train_org = X_train.copy()

        target = list(y_train.columns)
        cv_indices = src.evaluate_regression.custom_cross_validated_indices(pd.concat([X_train, y_train], axis=1),
                                                                            FACTORS,
                                                                            target,
                                                                            n_splits=cfg["modelling"]["k_fold"],
                                                                            shuffle=True,
                                                                            random_state=1444)

        # Preprocess
        X_train, scaler = src.encoding.ohe_encode_train_data(X_train=X_train,
                                                             cols_to_encode=["model", "tuning", "scoring"],
                                                             verbosity=verbosity)
        X_test = src.encoding.ohe_encode_test_data(X_test=X_test, cols_to_encode=["model", "tuning", "scoring"],
                                                   ohe=scaler, verbosity=verbosity)

        X_train = add_dataset_meta_information(df=X_train,
                                               path_to_meta_df=cfg["paths"]["dataset_meta_information_path"],
                                               nan_threshold=cfg["feature_engineering"]["dataset_meta_information"][
                                                   "nan_threshold"],
                                               replacing_strategy=
                                               cfg["feature_engineering"]["dataset_meta_information"][
                                                   "replacing_strategy"])
        X_test = add_dataset_meta_information(df=X_test,
                                              path_to_meta_df=cfg["paths"]["dataset_meta_information_path"],
                                              nan_threshold=cfg["feature_engineering"]["dataset_meta_information"][
                                                  "nan_threshold"],
                                              replacing_strategy=
                                              cfg["feature_engineering"]["dataset_meta_information"][
                                                  "replacing_strategy"])

        # Drop correlated features
        X_train, X_test = drop_pearson_correlated_features(train_data=X_train, test_data=X_test, threshold=0.7,
                                                           verbosity=verbosity)

        # Perform CV
        scores, histories = src.neural_net.perform_cv_neural_net(cfg, cv_indices, X_train, y_train,
                                                                 X_train_org, FACTORS, NEW_INDEX, verbosity)

        # Train Neural Net on whole data
        model, history = src.neural_net.fit_neural_net(cfg=cfg, X_train=X_train, y_train=y_train,
                                                       X_test=X_test, verbosity=verbosity)

        # Make Predictions
        src.neural_net.make_prediction(cfg=cfg, model=model, X_test=X_test, columns=list(y_train.columns),
                                       verbosity=verbosity)

    else:
        print(f"'{cfg['general']['method']}' is no valid option. ")
        print(f"End run!")

    # Track time for total runtime and display end of pipeline
    runtime = time.time() - start_time
    print(30 * "=" + f" PIPELINE FINISHED ({src.utils.display_runtime(runtime)}) " + 30 * "=")


if __name__ == "__main__":
    main()
