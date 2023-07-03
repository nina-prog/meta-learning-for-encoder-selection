import xgboost
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
#import lightgbm as lgb
#import catboost


from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from src.evaluate_regression import custom_spearmanr_scorer


def train_model(model=None, train_data=None, train_labels=None, hyperparam_grid=None, verbosity=1, k_fold=5,
                indices=None):
    """
    Function to perform cross validation and fit the final estimator.
    First selects the model based on the model: str parameter (e.g. "Dummy") and whether a hyperparam_grid is provided.
    Then performs cross validation on the given model and fits the final estimator on the whole data.
    Lastly, prints the results of each scorer in the CV results.

    :param model: String of the model to use, e.g. "Dummy"
    :param train_data: Train data, i.e. X_train
    :param train_labels: Labels of train data, i.e. y_train
    :param scoring: List of scoring functions
    :param hyperparam_grid: Dict of hyperparameters
    :param verbosity: Verbosity level, e.g. 1
    :param k_fold: Number of k_folds for CV, e.g. 5
    :param indices: Indices for custom CV returned from custom_cross_validated_indices() function

    :return: Fitted estimator and cv_results
    """

    # Keep string of model for printing reasons
    model_string = model

    # Initialize model with default parameters if no hyperparameter grid is given
    if hyperparam_grid is None:
        model_collection = {
            "Dummy": DummyRegressor(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
            "XGBoost": xgboost.XGBRegressor(random_state=42),
            #"LGBM": lgb.LGBMRegressor(random_state=42),
            #"CatBoost": catboost.CatBoostRegressor(random_state=42),
            "LinearRegression": LinearRegression()
        }
    else:
        model_collection = {
            # TBD: Define here models with hyperparam_grid as args
        }

    # Get model from model collection using the model parameter (str)
    try:
        model = model_collection[model]
    except KeyError:
        model = model_collection["Dummy"]
        print(f"Model '{model}' not found in model collection! Using default DummyRegressor.")

    # Perform CV
    if verbosity > 0: print(f"Performing CV with {k_fold} folds ...")
    scoring = {
        'spearman': custom_spearmanr_scorer,
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }
    cv_results = cross_validate(estimator=model, X=train_data, y=train_labels,
                                cv=indices, scoring=scoring, n_jobs=-1, return_train_score=True)

    # Fit final model on all train data
    if verbosity > 0: print(f"Fitting final model ({model_string}) ...")
    model.fit(X=train_data, y=train_labels)

    # Iterate through the provided scoring (list) in cv_results and print results
    if verbosity > 0:
        print("")
        for scorer in scoring:
            print(f"CV Training {scorer}: {round(cv_results[f'train_{scorer}'].mean(), 4)} "
                  f"+/- {round(cv_results[f'train_{scorer}'].std(), 4)} ")
            print(f"CV Test {scorer}: {round(cv_results[f'test_{scorer}'].mean(), 4)} "
                  f"+/- {round(cv_results[f'test_{scorer}'].std(), 4)}")
        print("")

    return model, cv_results


def make_prediction(model=None, test_data=None, result_path=None, save_data=True, target=None, verbosity=1):
    """
    Makes the prediction based on the input model and the test data.
    :param model: Fitted model
    :param test_data: Test data
    :param result_path: Path to save data
    :param save_data: bool -- Whether to save data in result_path or not
    :param target: str -- String of target column
    :param verbosity: Verbosity level

    :return: y_pred: pd.DataFrame of predicted values
    """

    # Make prediction and adjust index
    predictions = pd.DataFrame(model.predict(test_data), columns=[target + "_pred"])
    predictions.index = test_data.index

    # Save prediction and print information if save_data parm is True
    if save_data:
        predictions.to_csv(result_path, index=False)
        if verbosity > 0:
            print(f"Saved final prediction in '{result_path}'")

    return predictions
