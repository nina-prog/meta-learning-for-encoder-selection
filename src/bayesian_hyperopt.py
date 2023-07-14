"""
This python file is used to perform bayessearch hyperparameter optimization on the processed data.
The module is tested on pointwise classifier (week 11), i.e. predict the rank as a classifier problem.
It is not intended to call this file on the pipeline. Instead, execute this file using the follow commands
while in the root directory, i.e. phase-2:
python3 src/bayesian_hyperopt.py --config "configs/config.yaml"
For testing purposes, reduce sample size via:
python3 src/bayesian_hyperopt.py --config "configs/config.yaml" --subsample 10000

Troubleshooting:
- If there is an error with np.int deprecated, change the according line to np.int32 in the numpy source file
"""
import pandas as pd
import time
import os
import datetime

from sklearn.ensemble import ExtraTreesClassifier

from encoding import poincare_encoding, ohe_encode_train_data
from evaluate_regression import custom_spearmanr_scorer, custom_cross_validated_indices
from feature_engineering import normalize_train_data
from load_datasets import load_dataset
from meta_information import add_dataset_meta_information
from utils import parse_args, parse_config, display_runtime

from skopt import BayesSearchCV
from skopt.space import Integer


# Constants for supervisors functions
FACTORS = ["dataset", "model", "tuning", "scoring"]
NEW_INDEX = "encoder"
TARGET = "rank"


def perform_hyperopt(X_train, y_train, indices=None, n_iter=100):
    """ Performs BayesSearchCV.

    Defines the parameter grid and its model and calls the BayesSearchCV func from skopt.
    Then returns the fitted BayesSearchCV object (fitted model on all train data) and the cv_results.

    :param X_train: Train Data
    :param y_train: Labels of Train Data
    :param indices: Custom indices
    :param n_iter: int - Number of iterations for the BayesSearchCV

    :return: Fitted BayesSearchCV object and CV_results as pd.DataFrame
    """

    # Define base model to optimize
    model = ExtraTreesClassifier(random_state=42, n_jobs=-1)
    model_str = str(model).split("(")[0]

    # Define param grid
    param_grid = {'max_depth': Integer(3, 30),
                  'min_samples_split': Integer(2, 15),
                  'min_samples_leaf': Integer(2, 15),
                  'n_estimators': Integer(50, 800)
                  }

    start_time = time.time()
    print(f"Performing BayesSearchCV on {model_str} with {n_iter} Iterations ...")
    bayes_opt = BayesSearchCV(estimator=model, n_iter=n_iter,
                              search_spaces=param_grid,
                              n_jobs=-1, cv=indices, random_state=42,
                              scoring=custom_spearmanr_scorer,
                              return_train_score=True, verbose=1)
    bayes_opt.fit(X_train, y_train)

    # Print information about runtime
    print(f"\nBayesSearchCV took {(time.time() - start_time):.2f} seconds")

    # Transform CV Result to pandas for returning and sort ascending based on best mean test score
    bayes_opt_cv_results = pd.DataFrame(bayes_opt.cv_results_)
    bayes_opt_cv_results = bayes_opt_cv_results.sort_values(by="rank_test_score", ascending=True)
    # Get mean and std of test scores of each fold and print it
    mean = bayes_opt_cv_results['mean_test_score'].values[0]
    std = bayes_opt_cv_results['std_test_score'].values[0]

    # Print information
    print(f"Best Mean Test Score: {mean:.4f} +/- {std:.4f} average_spearman")
    print(f"Best Params: {dict(bayes_opt.best_params_)}\n")

    return bayes_opt, bayes_opt_cv_results


def save_cv_results(cv_results: pd.DataFrame, path: str) -> None:
    """
    Function to save csv file to specified folder
    :param cv_results: DataFrame to save
    :param path: Path to save cv results to
    :return: None
    """
    # Check if results folder exists, if not create one
    if not os.path.exists(path):
        print(f"{path} dir does not exist yet. Create one.")
        os.makedirs(path)

    # Save df as csv with unique name
    path = f"{path}bayesopt_cv_results_from_{str(datetime.datetime.now()).replace(':', '_')}.csv"
    cv_results.to_csv(path_or_buf=path)
    print(f"Saved CV Results in: '{path}'")


def get_processed_data():
    # Parse arguments and read config file cfg
    args = parse_args()
    cfg, cfg_path = parse_config(args)
    verbosity = cfg["general"]["verbosity"]

    ### LOAD DATA ###
    df_train = load_dataset(path=cfg["paths"]["train_rank_data_path"],
                            verbosity=verbosity,
                            subsample=args.subsample)
    # Drop cv_score
    if "cv_score" in df_train.columns:
        df_train = df_train.drop(columns=["cv_score"], axis=1)

    # Get indices for custom cross validation that is used within the modelling.py module
    indices = custom_cross_validated_indices(df_train,
                                             FACTORS, TARGET,
                                             n_splits=cfg["modelling"]["k_fold"],
                                             shuffle=True, random_state=1444)
    X_train = df_train.drop(TARGET, axis=1)
    y_train = df_train[TARGET]

    ### FEATURE ENGINEERING ###
    # General encodings: One Hot Encode (OHE) subset of features
    X_train, ohe = ohe_encode_train_data(X_train=X_train,
                                         cols_to_encode=cfg["feature_engineering"]["features_to_ohe"],
                                         verbosity=verbosity)

    # Encoder encoding: Poincare Embeddings for feature "encoder"
    X_train, _ = poincare_encoding(path_to_graph=cfg["paths"]["graph_path"],
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

    # Add dataset_agg (= csv-file containing meta information about the datasets)
    # The file can be created with the notebook from week 09
    X_train = add_dataset_meta_information(df=X_train,
                                           path_to_meta_df=cfg["paths"]["dataset_meta_information_path"],
                                           nan_threshold=cfg["feature_engineering"]["dataset_meta_information"][
                                               "nan_threshold"],
                                           replacing_strategy=cfg["feature_engineering"]["dataset_meta_information"][
                                               "replacing_strategy"])

    ### NORMALIZATION ###
    X_train, scaler = normalize_train_data(X_train=X_train, method=cfg["feature_engineering"]["normalize"]["method"],
                                           verbosity=verbosity)

    return X_train, y_train, indices


def main():
    """
    Main function to execute all necessary steps for the BayesSearchCV i.e.
    - Data Preprocessing
    - BayesSearchCV
    - Saving of the Results
    """

    start_time = time.time()
    print(30 * "=" + " STARTING BAYESSEARCHCV PIPELINE " + 30 * "=")

    # Get processed data
    X_train, y_train, indices = get_processed_data()

    # Perform BayesSearchCV
    bayes_opt, bayes_opt_cv_results = perform_hyperopt(X_train, y_train, indices=indices)

    # Save Dataframe
    save_cv_results(cv_results=bayes_opt_cv_results,
                    path="data/hyperparam_opt_results/")

    runtime = time.time() - start_time
    print(25 * "=" + f" BAYESSEARCHCV PIPELINE FINISHED ({display_runtime(runtime)}) " + 25 * "=")


# Execute from root dir via
# python3 src/bayesian_hyperopt.py --config "configs/config.yaml"
if __name__ == "__main__":
    main()
