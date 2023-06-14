import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


def train_model(model=None, train_data=None, train_labels=None, scoring=None, hyperparam_grid=None, verbosity=1, k_fold=5):
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

    :return: Fitted estimator and cv_results
    """

    # Keep string of model for printing reasons
    model_string = model

    # Initialize model with default parameters if no hyperparameter grid is given
    if hyperparam_grid is None:
        model_collection = {
            "Dummy": DummyRegressor(),
            "RandomForest": RandomForestRegressor(random_state=42),
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
    cv_results = cross_validate(estimator=model, X=train_data, y=train_labels,
                                cv=k_fold, scoring=scoring, n_jobs=-1, return_train_score=True)

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


# TODO: Function will be deprecated as soon as we the get train test data from our supervisor
def train_test_split_data(train_data, split_size=0.2):
    """
    Splits the overall data into train and test sets with fraction split_size.
    :param train_data: X_train
    :param train_labels: y_train
    :param split_size: Fraction of split_size

    :return: X_train, X_test, y_train, y_test
    """

    # Split data into features and target
    train_values = train_data.drop("cv_score", axis=1)
    train_labels = train_data["cv_score"]

    X_train, X_test, y_train, y_test = train_test_split(train_values, train_labels,
                                                        test_size = split_size, random_state = 42)
    return X_train, X_test, y_train, y_test


def make_prediction(model=None, test_data=None, result_path=None, verbosity=1):
    """
    Makes the prediction based on the input model and the test data.
    :param model: Fitted model
    :param test_data: Test data
    :param result_path: Path to save data
    :param verbosity: Verbosity level

    :return: None
    """

    # Make prediction
    predictions = pd.DataFrame(model.predict(test_data), columns=["cv_score"])
    predictions.to_csv(result_path, index=False)
    if verbosity > 0: print(f"Saved final prediction in '{result_path}'")
