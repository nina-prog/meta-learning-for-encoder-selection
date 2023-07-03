import pandas as pd

from pathlib import Path
from typing import Union


def load_dataset(path: Union[Path, str], verbosity=1, subsample=None) -> pd.DataFrame:
    if verbosity > 0:
        print(f"Loading data from '{path}' ...")
    # Read dataframe
    df = pd.read_csv(path, index_col=None)

    # Take subsample, if provided in the args
    if subsample is not None:
        print(f"Taking first {subsample} rows ...")
        df = df.iloc[:subsample]

    return df


def load_rankings(path: Union[Path, str], verbosity=1, subsample=None) -> pd.DataFrame:
    if verbosity > 0:
        print(f"Loading rankings from '{path}' ...")
    # Read dataframe
    out = pd.read_csv(path, index_col=0, header=[0, 1, 2, 3])
    out.columns.name = ("dataset", "model", "tuning", "scoring")

    # Take subsample, if provided in the args
    if subsample is not None:
        print(f"Taking first {subsample} rows ...")
        out = out.iloc[:subsample]


    return out


def load_train_data(path: Union[Path, str], verbosity=1, subsample=None) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads the Train data and splits into X_train and y_train

    :param path: str -- Path to the train data
    :param verbosity: int -- Level of verbosity
    :param subsample: int -- Number of subsamples to take

    :return: Tuple(pd.DataFrame, pd.DataFrame) -- X_train and y_train
    """

    if verbosity > 0:
        print(f"Loading train data from '{path}'...")

    # Read csv file
    df = pd.read_csv(path, index_col=None)

    # Take subsample, if provided in the args
    if subsample is not None:
        print(f"Taking first {subsample} rows ...")
        df = df.iloc[:subsample]

    # Split into features and target
    X_train = df.drop("cv_score", axis=1)
    y_train = df["cv_score"]

    return X_train, y_train


def load_test_data(path: Union[Path, str], verbosity=1, subsample=None) -> pd.DataFrame:
    """
    Loads the Test data

    :param path: str -- Path to the test data
    :param verbosity: int -- Level of verbosity
    :param subsample: int -- Number of subsamples to take

    :return: pd.DataFrame -- X_test
    """

    if verbosity > 0:
        print(f"Loading test data from '{path}'...")
    # Read dataframe
    df = pd.read_csv(path, index_col=None)

    # Take subsample, if provided in the args
    if subsample is not None:
        print(f"Taking first {subsample} rows ...")
        df = df.iloc[:subsample]

    return df

