""" This module contains the functions to encode the datasets into a graph and to embed the graph. """
import networkx as nx
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


# load data

# load graph

def load_graph(path: Union[Path, str]) -> nx.Graph:
    """
    Load a graph from a file. The file must be in the format of an adjacency list.
    
    :param path: Path to the file containing the graph.
    :type path: Union[Path, str]

    :return: The graph.
    """
    return nx.read_adjlist(path)


def ohe_encode_train_data(X_train: pd.DataFrame, cols_to_encode: list, verbosity=1) -> (pd.DataFrame, OneHotEncoder):
    """
    Function to One Hot Encode the train data: Fits and transforms the OHE Object on the train data;
    more specifically: The provided cols_to_encode (list of features). Function also makes sure that a
    pd.DataFrame is returned by dropping the old features and concatenating the encoded ones.

    :param X_train: pd.DataFrame -- Provided Train Dataset
    :param cols_to_encode: list -- Provided list of features to apply OHE on
    :param verbosity: int -- Level of verbosity

    :return: Tuple with pd.DataFrame with encoded features and fitted OHE object
    """
    if verbosity > 0:
        print(f"One Hot Encoding the features {cols_to_encode} of the train data ...")

    # Get DataFrame with only relevant features, i.e. cols_to_encode
    X_train_cats = X_train[cols_to_encode]

    # Fit OneHotEncoding object
    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    X_train_cats_encoded = ohe.fit_transform(X_train_cats).toarray()

    # Transform encoded data to pandas dataframe
    X_train_cats_encoded = pd.DataFrame(X_train_cats_encoded, columns=ohe.get_feature_names_out(), index=X_train.index)

    # Drop old features
    feats_to_drop = list(ohe.feature_names_in_)
    X_train = X_train.drop(columns=feats_to_drop, axis=1)

    # Concat old dataframe with new encoded features
    X_train_encoded = pd.concat([X_train, X_train_cats_encoded], axis=1)

    return X_train_encoded, ohe


def ohe_encode_test_data(X_test: pd.DataFrame, cols_to_encode: list, ohe: OneHotEncoder, verbosity=1) -> pd.DataFrame:
    """
    Function to apply the fitted OHE object on the test set features provided in param cols_to_encode.
    Also makes sure that pd.DataFrame is returned by dropping the old features and concatenating the encoded ones.

    :param X_test: pd.DataFrame -- Provided Test Dataset
    :param cols_to_encode: list -- Provided list of features to apply OHE on
    :param ohe: OneHotEncoder -- Fitted OHE object
    :param verbosity: int -- Level of verbosity

    :return: pd.DataFrame -- Encoded Test Dataset
    """
    if verbosity > 0:
        print(f"One Hot Encoding the features {cols_to_encode} of the test data ...")

    # Get DataFrame with only relevant features, i.e. cols_to_encode and transform them
    X_test_cats = X_test[cols_to_encode]
    X_test_cats_encoded = ohe.transform(X_test_cats).toarray()

    # Transform to pandas DataFrame
    X_test_cats_encoded = pd.DataFrame(X_test_cats_encoded, columns=ohe.get_feature_names_out(), index=X_test.index)

    # Drop old features
    feats_to_drop = list(ohe.feature_names_in_)
    X_test = X_test.drop(columns=feats_to_drop, axis=1)

    # Concat old dataframe with new encoded features
    X_test_encoded = pd.concat([X_test, X_test_cats_encoded], axis=1)

    return X_test_encoded