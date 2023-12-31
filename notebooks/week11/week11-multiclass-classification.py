import os
import pandas as pd
import numpy as np
import math

import sys
sys.path.append("../..")
from src.load_datasets import load_dataset, load_rankings, load_train_data
import src.evaluate_regression

import src.encoding
from src.feature_engineering import normalize_train_data, normalize_test_data
from src.meta_information import add_dataset_meta_information
from src.evaluate_regression import custom_spearmanr_scorer

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
import xgboost


def get_pearson_correlated_features(data=None, threshold=0.7):
    """
    Calculates the pearson correlation of all features in the dataframe and returns a set of features with a
    correlation greater than the threshold.

    :param data: The input dataframe.
    :type data: pd.DataFrame
    :param threshold: The threshold for the correlation coefficient in the range of [0.0, 1.0].
    :type threshold: float,optional(default=0.7)

    :return: The set of features with a correlation greater than the threshold.
    :rtype: set
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    return correlated_features


# Define variables for ranking
factors = ["dataset", "model", "tuning", "scoring"]
new_index = "encoder"
target = "rank"

# Load data
df_train = load_dataset("./data/raw/dataset_rank_train.csv")

if "cv_score" in df_train.columns:
    df_train = df_train.drop("cv_score", axis=1)

X_train = df_train.drop(target, axis=1)
y_train = df_train[target]


# Create indices for cv
cv_indices = src.evaluate_regression.custom_cross_validated_indices(pd.concat([X_train, y_train], axis=1), 
                                                                    factors, 
                                                                    target, 
                                                                    n_splits=5, 
                                                                    shuffle=True, 
                                                                    random_state=1444)
# Preprocessing
# OHE encoding 
X_train, ohe = src.encoding.ohe_encode_train_data(X_train=X_train,
                                                  cols_to_encode=["model", "tuning", "scoring"],
                                                  verbosity=2)

# Encoder encoding: Poincare Embeddings for feature "encoder"
X_train, _ = src.encoding.poincare_encoding(path_to_graph="./data/raw/graph.adjlist",
                                            path_to_embeddings="./data/preprocessed/embeddings.csv",
                                            data=X_train,
                                            column_to_encode="encoder",
                                            encode_dim=15,
                                            explode_dim=True,
                                            epochs=5000,
                                            dim_reduction=None,
                                            verbosity=2)

# Add meta information
X_train = add_dataset_meta_information(df=X_train,
                                       path_to_meta_df="./data/preprocessed/dataset_agg.csv",
                                       nan_threshold=0.4,
                                       replacing_strategy="median")

# Normalize data
X_train, scaler = normalize_train_data(X_train=X_train, 
                                       method="minmax",
                                       verbosity=2)

# Get correlated features
print("Drop correlated features...")
correlated_features = get_pearson_correlated_features(data=X_train)
#print(f"Correlated features: {correlated_features}")

# Filter out some features
correlated_features = [f for f in correlated_features if not f.startswith("enc_dim_")]
correlated_features = [f for f in correlated_features if not f.startswith("model_")]
correlated_features = [f for f in correlated_features if not f.startswith("tuning_")]
correlated_features = [f for f in correlated_features if not f.startswith("scoring_")]

# Drop features
X_train = X_train.drop(correlated_features, axis=1)


# Feature selection
print("Feature selection...")
fs = SelectKBest(score_func=f_regression, k='all')  # or mutual_info_regression
fs.fit(X_train, y_train.ravel())

# Select columns based on mask
mask = [x >= np.quantile(fs.scores_, 0.4) for x in fs.scores_]  # 0.4
X_train_fs = X_train.loc[:, mask]
selected_features = list(X_train_fs.columns)
sf = list(X_train.columns)
sf = [f for f in sf if f not in selected_features or not f.startswith("enc_dim_") or not f.startswith("tuning_") or not f.startswith("scoring_") or not f.startswith("model_")]
print(sf)
#print(selected_features)

X_train = X_train[sf]

# Classification
# Use the labels as they are
# Define models
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
dt = DecisionTreeClassifier(random_state=42)
et = ExtraTreeClassifier(random_state=42)
ets = ExtraTreesClassifier(random_state=42, n_jobs=-1)
knn = KNeighborsClassifier(n_jobs=-1)
svc = LinearSVC(random_state=42, multi_class="crammer_singer")
rnc = RadiusNeighborsClassifier(n_jobs=-1, radius=5)
gpc = GaussianProcessClassifier(random_state=42, multi_class="one_vs_rest")  # "one_vs_one"
# XGBoost from phase-1
xgb = xgboost.XGBClassifier(colsample_bytree=0.27972729119255346,
                            learning_rate=0.1228007619140701,
                            max_depth=23,
                            n_estimators=144,
                            reg_alpha=1e-09,
                            reg_lambda=18.935672936151313,
                            subsample=1.0,
                            random_state=42,
                            n_jobs=-1)

# Remove other models, because of performance
models = [rf, dt, et, ets, knn]

scoring = {
    'spearman': custom_spearmanr_scorer,
    'MCC'     : make_scorer(matthews_corrcoef)
}


# ToDo: More models
# Traverse models and score
for model in models:
    print(model)
    cv_results = cross_validate(estimator=model, 
                                X=X_train,
                                y=y_train,
                                cv=cv_indices, 
                                scoring=scoring,
                                n_jobs=-1, 
                                return_train_score=True)

    for scorer in list(scoring.keys()):
        print(f"CV Test {scorer}: \t{round(cv_results[f'test_{scorer}'].mean(), 4)} +/- {round(cv_results[f'test_{scorer}'].std(), 4)}")
    print("")
