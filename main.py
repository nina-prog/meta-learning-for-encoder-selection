"""
Module for the pipeline.
Execute via python3 main.py --config configs/config.yaml
or if you want to run pipeline with subsample of size 100
python3 main.py --config configs/config.yaml --subsample 100
"""
import time
import pandas as pd

import mlflow.sklearn

import src.utils
import src.load_datasets
import src.modelling
import src.mlflow_registry
import src.encoding
import src.evaluate_regression

from src.feature_engineering import normalize_train_data, normalize_test_data
from src.meta_information import add_dataset_meta_information

# Constants for supervisors functions
FACTORS = ["dataset", "model", "tuning", "scoring"]
NEW_INDEX = "encoder"
TARGET = "rank"


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
    # Drop cv_score
    df_train = df_train.drop(columns=["cv_score"], axis=1)

    # Split into train and validation set for internal comparison / evaluation
    X_train, X_holdout, y_train, y_holdout = src.evaluate_regression.custom_train_test_split(df_train, FACTORS, TARGET)
    # Load Test data (Hold-out-set) that is used for course-internal scoring
    X_test = src.load_datasets.load_test_data(path=cfg["paths"]["test_values_path"],
                                              verbosity=verbosity,
                                              subsample=args.subsample)
    # Make copy of validation set to fit into schema that supervisor function get_rankings() expects
    X_holdout_original = X_holdout.copy()
    # Get indices for custom cross validation that is used within the modelling.py module
    indices = src.evaluate_regression.custom_cross_validated_indices(pd.concat([X_train, y_train], axis=1),
                                                                     FACTORS, TARGET,
                                                                     n_splits=cfg["modelling"]["k_fold"],
                                                                     shuffle=True, random_state=1444)

    ### FEATURE ENGINEERING ###
    # General encodings: One Hot Encode (OHE) subset of features
    X_train, ohe = src.encoding.ohe_encode_train_data(X_train=X_train,
                                                      cols_to_encode=cfg["feature_engineering"]["features_to_ohe"],
                                                      verbosity=verbosity)
    X_holdout = src.encoding.ohe_encode_test_data(X_test=X_holdout,
                                                  cols_to_encode=cfg["feature_engineering"]["features_to_ohe"],
                                                  ohe=ohe, verbosity=verbosity)
    X_test = src.encoding.ohe_encode_test_data(X_test=X_test,
                                               cols_to_encode=cfg["feature_engineering"]["features_to_ohe"],
                                               ohe=ohe, verbosity=verbosity)

    # Encoder encoding: Poincare Embeddings for feature "encoder"
    X_train, _ = src.encoding.poincare_encoding(path_to_graph=cfg["paths"]["graph_path"],
                                                path_to_embeddings=cfg["paths"]["embeddings_path"],
                                                data=X_train,
                                                column_to_encode="encoder",
                                                encode_dim=cfg["feature_engineering"]["poincare_embedding"]["dim"],
                                                explode_dim=cfg["feature_engineering"]["poincare_embedding"][
                                                    "explode_dim"],
                                                epochs=cfg["feature_engineering"]["poincare_embedding"][
                                                    "epochs"],
                                                dim_reduction=cfg["feature_engineering"]["poincare_embedding"][
                                                    "dim_reduction"],
                                                verbosity=verbosity)
    X_holdout, _ = src.encoding.poincare_encoding(path_to_embeddings=cfg["paths"]["embeddings_path"],
                                                  data=X_holdout,
                                                  column_to_encode="encoder",
                                                  explode_dim=cfg["feature_engineering"]["poincare_embedding"][
                                                      "explode_dim"],
                                                  verbosity=verbosity)
    X_test, _ = src.encoding.poincare_encoding(path_to_embeddings=cfg["paths"]["embeddings_path"],
                                               data=X_test,
                                               column_to_encode="encoder",
                                               explode_dim=cfg["feature_engineering"]["poincare_embedding"][
                                                   "explode_dim"],
                                               verbosity=verbosity)

    # Add dataset_agg (= csv-file containing meta information about the datasets)
    # The file can be created with the notebook from week 09
    X_train = add_dataset_meta_information(df=X_train,
                                           path_to_meta_df=cfg["paths"]["dataset_meta_information_path"],
                                           nan_threshold=cfg["feature_engineering"]["dataset_meta_information"][
                                               "nan_threshold"],
                                           replacing_strategy=cfg["feature_engineering"]["dataset_meta_information"][
                                               "replacing_strategy"])
    X_holdout = add_dataset_meta_information(df=X_holdout,
                                             path_to_meta_df=cfg["paths"]["dataset_meta_information_path"],
                                             nan_threshold=cfg["feature_engineering"]["dataset_meta_information"][
                                                 "nan_threshold"],
                                             replacing_strategy=cfg["feature_engineering"]["dataset_meta_information"][
                                                 "replacing_strategy"])
    X_test = add_dataset_meta_information(df=X_test,
                                          path_to_meta_df=cfg["paths"]["dataset_meta_information_path"],
                                          nan_threshold=cfg["feature_engineering"]["dataset_meta_information"][
                                              "nan_threshold"],
                                          replacing_strategy=cfg["feature_engineering"]["dataset_meta_information"][
                                              "replacing_strategy"])

    ### NORMALIZATION ###
    X_train, scaler = normalize_train_data(X_train=X_train, method=cfg["feature_engineering"]["normalize"]["method"],
                                           verbosity=verbosity)
    X_holdout = normalize_test_data(X_test=X_holdout, scaler=scaler, verbosity=verbosity)
    X_test = normalize_test_data(X_test=X_test, scaler=scaler, verbosity=verbosity)

    # Save processed data in data/preprocessed
    print("Saving processed data in 'data/preprocessed/' ...")
    X_holdout_original.to_csv("data/preprocessed/X_hold_out_original.csv", index=False)
    X_holdout.to_csv("data/preprocessed/X_holdout.csv", index=False)
    y_holdout.to_csv("data/preprocessed/y_holdout.csv", index=False)
    X_train.to_csv("data/preprocessed/X_train.csv", index=False)
    y_train.to_csv("data/preprocessed/y_train.csv", index=False)

    ### MODELLING ###
    # Log model evaluation to mlflow registry
    mlflow.sklearn.autolog(log_models=False)
    with mlflow.start_run(tags=src.mlflow_registry.get_mlflow_tags(X_train, cfg)) as run:
        # Perform CV and train model
        model, cv_result = src.modelling.train_model(model=cfg["modelling"]["model"],
                                                     train_data=X_train,
                                                     train_labels=y_train,
                                                     hyperparam_grid=None,
                                                     verbosity=verbosity,
                                                     k_fold=cfg["modelling"]["k_fold"],
                                                     indices=indices)
        # Log additional information to mlflow run
        src.mlflow_registry.log_model_eval(cv_result, cfg_path, run, verbosity)

        ### PREDICTIONS ###
        # Make final predictions on test data and validation data
        y_pred_holdout = src.modelling.make_prediction(model=model, test_data=X_holdout,
                                                       result_path=cfg["paths"]["result_path"], save_data=False,
                                                       target=TARGET, verbosity=verbosity)

        y_pred_test = src.modelling.make_prediction(model=model, test_data=X_test,
                                                    result_path=cfg["paths"]["result_path"], save_data=True,
                                                    target=TARGET, verbosity=verbosity)

        ### RANKINGS ###
        # Concat to df_pred for spearman evaluation
        df_pred = pd.concat([X_holdout_original, y_holdout, y_pred_holdout], axis=1)
        rankings_test = src.evaluate_regression.get_rankings(df_pred, factors=FACTORS,
                                                             new_index=NEW_INDEX, target=TARGET)
        rankings_pred = src.evaluate_regression.get_rankings(df_pred, factors=FACTORS,
                                                             new_index=NEW_INDEX, target=TARGET+"_pred")

        # Get Average Spearman, print and log it
        avg_spearman = src.evaluate_regression.average_spearman(rankings_test, rankings_pred)
        print(f"Average Spearman of hold-out-set: {avg_spearman:.4f}")
        mlflow.log_metric(key="average_spearman", value=avg_spearman)

    # Track time for total runtime and display end of pipeline
    runtime = time.time() - start_time
    print(30 * "=" + f" PIPELINE FINISHED ({src.utils.display_runtime(runtime)}) " + 30 * "=")


if __name__ == "__main__":
    main()
