"""
Evaluation script to evaluate the regression task.

Example usage to evaluate a pipeline:

import pandas as pd

from category_encoders import OneHotEncoder
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.load_datasets import load_dataset

factors = ["dataset", "model", "tuning", "scoring"]
new_index = "encoder"
target = "cv_score"

# ---- load data...
DATA_DIR = Path(".\data")

df_train = load_dataset(DATA_DIR / "dataset_train.csv")
X_train = df_train.drop("cv_score", axis=1)
y_train = df_train["cv_score"]
df_test = load_dataset(DATA_DIR / "dataset_test.csv")  # you do not have this dataset
X_test = df_test.drop("cv_score", axis=1)
y_test = df_test["cv_score"]

# ---- ... or split a dataset
# df_train = load_dataset(DATA_DIR / "dataset_train.csv")
# X_train, X_test, y_train, y_test = custom_train_test_split(df, factors, target)

# ---- predict ...
dummy_pipe = Pipeline([("encoder", OneHotEncoder()), ("model", DecisionTreeRegressor())])
y_pred = pd.Series(dummy_pipe.fit(X_train, y_train).predict(X_test), index=y_test.index, name="cv_score_pred")
df_pred = pd.concat([X_test, y_test, y_pred], axis=1)

# ---- ... or tune
dummy_pipe = Pipeline([("encoder", OneHotEncoder()), ("model", DecisionTreeRegressor())])
indices = er.custom_cross_validated_indices(df_train, factors, target, n_splits=5, shuffle=True, random_state=1444)

gs = GridSearchCV(dummy_pipe, {"model__max_depth": [2, 5, None]}, cv=indices).fit(X_train, y_train)
print(gs.best_params_)

# ---- convert to rankings and evaluate
rankings_test = get_rankings(df_pred, factors=factors, new_index=new_index, target="cv_score")
rankings_pred = get_rankings(df_pred, factors=factors, new_index=new_index, target="cv_score_pred")
print(average_spearman(rankings_test, rankings_pred))
"""

import numpy as np
import pandas as pd
import warnings

from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.model_selection import train_test_split, KFold
from typing import Iterable, List


def custom_train_test_split(df: pd.DataFrame, factors, target, train_size=0.75, shuffle=True, random_state=0):
    """
    Like train_test_split, but keeps encoders together: the split is on 'factors'.
    """

    X_factors = df.groupby(factors).agg(lambda a: np.nan).reset_index()[factors]
    factors_train, factors_test = train_test_split(X_factors, stratify=X_factors.dataset,
                                                   train_size=train_size, shuffle=shuffle, random_state=random_state)

    df_train = pd.merge(factors_train, df, on=factors)
    df_test = pd.merge(factors_test, df, on=factors)

    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]

    return X_train, X_test, y_train, y_test


def custom_cross_validated_indices(df: pd.DataFrame, factors: Iterable[str], target: str,
                                   **kfoldargs) -> List[List[Iterable[int]]]:
    df_factors = df.groupby(factors)[target].mean().reset_index()
    X_factors, y_factors = df_factors.drop(target, axis=1), df_factors[target]

    indices = []
    for itr, ite in KFold(**kfoldargs).split(X_factors, y_factors):
        tr = pd.merge(X_factors.iloc[itr], df.reset_index(), on=factors).index  # "index" is the index of df
        te = pd.merge(X_factors.iloc[ite], df.reset_index(), on=factors).index  # "index" is the index of df
        indices.append([tr, te])

    return indices


def score2ranking(score: pd.Series, ascending=True):
    """
    Ascending =
        True: lower score = better rank (for instance, if score is the result of a loss function or a ranking itself)
        False: greater score = better rank (for instance, if score is the result of a score such as roc_auc_score)
    """
    c = 1 if ascending else -1
    order_map = {
        s: sorted(score.unique(), key=lambda x: c * x).index(s) for s in score.unique()
    }
    return score.map(order_map)


def get_rankings(df: pd.DataFrame, factors, new_index, target) -> pd.DataFrame:

    assert set(factors).issubset(df.columns)

    rankings = {}
    for group, indices in df.groupby(factors).groups.items():
        score = df.iloc[indices].set_index(new_index)[target]
        rankings[group] = score2ranking(score, ascending=False)

    return pd.DataFrame(rankings)


def spearman_rho(x: Iterable, y: Iterable, nan_policy="omit"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        return spearmanr(x, y, nan_policy=nan_policy)[0]


def list_spearman(rf1: pd.DataFrame, rf2: pd.DataFrame) -> np.array:
    if not rf1.columns.equals(rf2.columns) or not rf1.index.equals(rf2.index):
        raise ValueError("The two input dataframes should have the same index and columns.")

    return np.array([
        spearman_rho(r1, r2, nan_policy="omit") for (_, r1), (_, r2) in zip(rf1.items(), rf2.items())
    ])


def average_spearman(rf1: pd.DataFrame, rf2: pd.DataFrame) -> np.array:
    return np.nanmean(list_spearman(rf1, rf2))







