import time
import pandas as pd
import numpy as np

import mlflow.sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import src.utils
import src.load_datasets
import src.modelling
import src.mlflow_registry
import src.encoding
import src.evaluate_regression
import src.data_cleaning

from src.feature_engineering import normalize_train_data, normalize_test_data
from src.meta_information import add_dataset_meta_information

import src.encoder_utils as eu
import src.evaluate_regression as er
import src.load_datasets as ld
import src.pairwise_utils as pu


# Constants for supervisors functions
FACTORS = ["dataset", "model", "tuning", "scoring"]
NEW_INDEX = "encoder"
TARGET = "rank"


def preprocessing(df_train, X_test, config):
    verbosity = config["general"]["verbosity"]
    
    # Drop cv_score
    if "cv_score" in df_train.columns:
        df_train = df_train.drop(columns=["cv_score"], axis=1)
    
    X_train = df_train[FACTORS + ["encoder"]].groupby(FACTORS).agg(lambda x: np.nan).reset_index()[FACTORS]
    y_train = pd.merge(X_train,
                       pu.get_pairwise_target(df_train, features=FACTORS, target=TARGET, column_to_compare="encoder"),
                       on=FACTORS, how="left").drop(FACTORS, axis=1).fillna(0)
    base_df = df_train.copy()

    
    # General encodings: One Hot Encode (OHE) subset of features
    X_train, ohe = src.encoding.ohe_encode_train_data(X_train=X_train,
                                                      cols_to_encode=config["feature_engineering"]["features_to_ohe"],
                                                      verbosity=verbosity)
    X_test = src.encoding.ohe_encode_test_data(X_test=X_test,
                                               cols_to_encode=config["feature_engineering"]["features_to_ohe"],
                                               ohe=ohe, 
                                               verbosity=verbosity)
    base_df_mod = src.encoding.ohe_encode_test_data(X_test=base_df,
                                                cols_to_encode=config["feature_engineering"]["features_to_ohe"],
                                                ohe=ohe, 
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
    
    return X_train, y_train, X_test, base_df_mod


def modelling(X_train, y_train, base_df, indices, config_path, config):
    verbosity = config["general"]["verbosity"]
    if verbosity > 0:
        print(f"Performing CV with {len(indices)} folds ...")

    # new factors for supervisor function
    new_factors = list(base_df.columns)
    new_factors.remove("rank")
    new_factors.remove("encoder")
    
    # Setup pipeline / model
    if config["feature_engineering"]["normalize"]["method"] == "minmax":
        scaler = MinMaxScaler()

    # Get model
    model, model_string = src.modelling.get_model(model=config["modelling"]["model"], 
                                                  hyperparam_grid=None)

    pipeline = Pipeline([("scaler", scaler), ("model", model)])
    
    # Iterate over folds
    #with mlflow.start_run(tags=src.mlflow_registry.get_mlflow_tags(X_train, config)) as run:
    scores = []
    for fold in indices:
        # Get train and test data for current split
        X_tr = X_train.iloc[fold[0]].copy()
        X_te = X_train.iloc[fold[1]].copy()
        y_tr = y_train.iloc[fold[0]].copy()
        y_te = y_train.iloc[fold[1]].copy()

        # Train and predict
        fitted_model = pipeline.fit(X_tr, y_tr)
        y_pred = pd.DataFrame(fitted_model.predict(X_te), columns=y_tr.columns, index=X_te.index)

        # Evaluate with given functions
        tmp = pu.join_pairwise2rankings(X_te, y_pred, list(X_train.columns))
        df_pred = pd.merge(base_df,
                           tmp,
                           on=new_factors + ["encoder"], 
                           how="inner")
        rankings_test = er.get_rankings(df_pred, factors=new_factors, new_index=NEW_INDEX, target="rank")
        rankings_pred = er.get_rankings(df_pred, factors=new_factors, new_index=NEW_INDEX, target="rank_pred")
        scores.append(er.average_spearman(rankings_test, rankings_pred))
        
        # Log additional information to mlflow run
        #src.mlflow_registry.log_model_eval(scores, config_path, run, verbosity)
    
    # Print results
    print(f"CV Test spearman: {round(np.mean(scores), 4)} +/- {round(np.std(scores), 4)}")
    
    # Fit final model on all train data
    if verbosity > 0:
        print(f"Fitting final model ({model_string}) ...")
    pipeline.fit(X=X_train, y=y_train)
    
    return pipeline


def prediction_pointwise(model, X_test, target_columns, config):
    verbosity = config["general"]["verbosity"]
    
    predictions = pd.DataFrame(model.predict(X_test), columns=target_columns)
    predictions.index = X_test.index

    # Save prediction and print information if save_data parm is True
    predictions.to_csv(config["paths"]["result_path"], 
                       index=False)
    if verbosity > 0:
        print(f"Saved final prediction in '{config['paths']['result_path']}'")
    
