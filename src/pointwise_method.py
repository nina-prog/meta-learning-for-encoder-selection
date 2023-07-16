import time
import pandas as pd

import mlflow.sklearn

import src.utils
import src.load_datasets
import src.modelling
import src.mlflow_registry
import src.encoding
import src.evaluate_regression
import src.data_cleaning

from src.feature_engineering import normalize_train_data, normalize_test_data
from src.meta_information import add_dataset_meta_information


# Constants for supervisors functions
FACTORS = ["dataset", "model", "tuning", "scoring"]
NEW_INDEX = "encoder"
TARGET = "rank"


def preprocessing(df_train, X_test, config):
    verbosity = config["general"]["verbosity"]
    
    # Drop cv_score
    if "cv_score" in df_train.columns:
        df_train = df_train.drop(columns=["cv_score"], axis=1)

    X_train = df_train.drop(TARGET, axis=1)
    y_train = df_train[[TARGET]]

    ### FEATURE ENGINEERING ###
    # General encodings: One Hot Encode (OHE) subset of features
    X_train, ohe = src.encoding.ohe_encode_train_data(X_train=X_train,
                                                      cols_to_encode=config["feature_engineering"]["features_to_ohe"],
                                                      verbosity=verbosity)
    X_test = src.encoding.ohe_encode_test_data(X_test=X_test,
                                               cols_to_encode=config["feature_engineering"]["features_to_ohe"],
                                               ohe=ohe, verbosity=verbosity)

    # Encoder encoding: Poincare Embeddings for feature "encoder"
    X_train, _ = src.encoding.poincare_encoding(path_to_graph=config["paths"]["graph_path"],
                                                path_to_embeddings=config["paths"]["embeddings_path"],
                                                data=X_train,
                                                column_to_encode="encoder",
                                                encode_dim=config["feature_engineering"]["poincare_embedding"]["dim"],
                                                explode_dim=config["feature_engineering"]["poincare_embedding"][
                                                    "explode_dim"],
                                                epochs=config["feature_engineering"]["poincare_embedding"][
                                                    "epochs"],
                                                dim_reduction=config["feature_engineering"]["poincare_embedding"][
                                                    "dim_reduction"],
                                                verbosity=verbosity)
    X_test, _ = src.encoding.poincare_encoding(path_to_embeddings=config["paths"]["embeddings_path"],
                                               data=X_test,
                                               column_to_encode="encoder",
                                               explode_dim=config["feature_engineering"]["poincare_embedding"][
                                                   "explode_dim"],
                                               verbosity=verbosity)

    # Add dataset_agg (= csv-file containing meta information about the datasets)
    # The file can be created with the notebook from week 09
    X_train = add_dataset_meta_information(df=X_train,
                                           path_to_meta_df=config["paths"]["dataset_meta_information_path"],
                                           nan_threshold=config["feature_engineering"]["dataset_meta_information"][
                                               "nan_threshold"],
                                           replacing_strategy=config["feature_engineering"]["dataset_meta_information"][
                                               "replacing_strategy"])
    X_test = add_dataset_meta_information(df=X_test,
                                          path_to_meta_df=config["paths"]["dataset_meta_information_path"],
                                          nan_threshold=config["feature_engineering"]["dataset_meta_information"][
                                              "nan_threshold"],
                                          replacing_strategy=config["feature_engineering"]["dataset_meta_information"][
                                              "replacing_strategy"])
    
    # Drop correlated features
    X_train, X_test = src.data_cleaning.drop_pearson_correlated_features(train_data=X_train, 
                                                                         test_data=X_test, 
                                                                         threshold=config["data_cleaning"]["pearson_correlation"]["threshold"], 
                                                                         verbosity=verbosity)
    
    # Select features 
    X_train, X_test = src.feature_engineering.feature_selection(X_train=X_train,
                                                                X_test=X_test,
                                                                y_train=y_train,
                                                                quantile=0.4,
                                                                verbosity=verbosity)
    
    # Normalization
    X_train, scaler = normalize_train_data(X_train=X_train, method=config["feature_engineering"]["normalize"]["method"],
                                           verbosity=verbosity)
    X_test = normalize_test_data(X_test=X_test, 
                                 scaler=scaler, 
                                 verbosity=verbosity)
    
    return X_train, y_train, X_test


def modelling(X_train, y_train, indices, config_path, config):
    verbosity = config["general"]["verbosity"]
    mlflow.sklearn.autolog(log_models=False)
    with mlflow.start_run(tags=src.mlflow_registry.get_mlflow_tags(X_train, config)) as run:
        # Perform CV and train model
        model, cv_result = src.modelling.train_model(model=config["modelling"]["model"],
                                                     train_data=X_train,
                                                     train_labels=y_train,
                                                     hyperparam_grid=None,
                                                     verbosity=verbosity,
                                                     k_fold=config["modelling"]["k_fold"],
                                                     indices=indices)
        # Log additional information to mlflow run
        src.mlflow_registry.log_model_eval(cv_result, config_path, run, verbosity)

    return model
    
        
def prediction_pointwise(model, X_test, config):
    verbosity = config["general"]["verbosity"]
    _ = src.modelling.make_prediction(model=model, 
                                      test_data=X_test,
                                      result_path=config["paths"]["result_path"], 
                                      save_data=True,
                                      target=TARGET, 
                                      verbosity=verbosity)
    
