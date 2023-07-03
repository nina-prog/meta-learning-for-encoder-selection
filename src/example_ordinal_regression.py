import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder

from src import load_datasets as ld, evaluate_regression as er, encoder_utils as eu

DATA_DIR = Path(r".\data")

factors = ["dataset", "model", "tuning", "scoring"]
new_index = "encoder"

df_train = ld.load_dataset(DATA_DIR / "dataset_rank_train.csv")
X_train, X_test, y_train, y_test = er.custom_train_test_split(df_train, factors, "rank")

# transform into multi-output binary classification problem
y_train = pd.DataFrame(y_train)
for i in range(int(y_train["rank"].max())):
    y_train[f"y{i}"] = (y_train["rank"] > i).astype(int)
y_train.drop("rank", axis=1, inplace=True)

"""
category_encoders.OneHotEncoder throws errors if y has more than one column --- no multi-output support. 
Solutions: 
    1. pre-process X_train, X_test separately and then train and test the model;
    2. wrap the encoder and do not pass y to it --- usable in a Pipeline.
"""
dummy_pipe = Pipeline([("encoder", eu.NoY(OneHotEncoder())), ("model", DecisionTreeClassifier())])

""" 
The predicted rank is:
    1. the sum of the predicted binary variables (rank_pred_sum);
    2. the maximum column index of the targets such that the column is 1, that is, the minimum index of 0 == min(y_pred) 
        (rank_pred_min).
"""
y_pred = dummy_pipe.fit(X_train, y_train).predict(X_test)

X_test["rank"] = y_test
X_test["rank_pred_sum"] = y_pred.sum(axis=1)
X_test["rank_pred_min"] = np.argmin(y_pred, axis=1) - 1

rankings_test = er.get_rankings(X_test, factors=factors, new_index=new_index, target="rank")
rankings_pred_sum = er.get_rankings(X_test, factors=factors, new_index=new_index, target="rank_pred_sum")
rankings_pred_min = er.get_rankings(X_test, factors=factors, new_index=new_index, target="rank_pred_min")

print("Ordinal regression average spearman:\n",
      f"Sum: {er.average_spearman(rankings_test, rankings_pred_sum):.03f}\n",
      f"Min: {er.average_spearman(rankings_test, rankings_pred_min):.03f}")


