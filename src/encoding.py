""" This module contains the functions to encode the datasets into a graph and to embed the graph. """
import networkx as nx
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from gensim.models.poincare import PoincareModel


def load_graph(path: Union[Path, str]) -> nx.Graph:
    """
    Load a graph from a file. The file must be in the format of an adjacency list.
    
    :param path: Path to the file containing the graph.
    :type path: Union[Path, str]

    :return: The graph.
    """
    G = nx.read_adjlist(path)

    # add node names as labels to the graph
    node_names = {node: node for node in G.nodes()}
    nx.set_node_attributes(G, node_names, "label")

    return G


def poincare_encoding(path_to_graph: str, data=None, column_to_encode=None, encode_dim=50, epochs=500, seed=7,
                      explode_dim=True, verbosity=1) -> Union[pd.DataFrame, tuple[pd.DataFrame, PoincareModel]]:
    """
    Generates the Poincarè embedding for the given graph and encodes the given column of the given data with it. The
    encoding can be done in different formats. The function can also be used to just generate the embedding for the
    given graph. The graph has to be given as an edge list.

    :param path_to_graph: Path to the graph.
    :type path_to_graph: str
    :param data: Data to encode.
    :type data: pandas.DataFrame
    :param column_to_encode: Column to encode.
    :type column_to_encode: str
    :param encode_dim: Dimension of the embedding.
    :type encode_dim: int
    :param epochs: Number of epochs to train the model.
    :type epochs: int
    :param seed: Seed for the random number generator.
    :type seed: int
    :param explode_dim: If True, the embedding is exploded into multiple columns.
    :type explode_dim: bool

    :return: The encoded data.
    :rtype: pandas.DataFrame
    """
    # load Graph
    G = load_graph(path_to_graph)

    # Embed the graph
    if verbosity > 0:
        print("(Poincare) Embedding the graph...")
    model = PoincareModel(list(G.edges()), seed=seed, size=encode_dim)
    model.train(epochs=epochs, print_every=500)

    # Get the embeddings and map them to the node names
    embeddings_dict = {node: model.kv[node] for node in G.nodes}
    emb_df = pd.DataFrame.from_dict(embeddings_dict, orient='index')

    if data is None or column_to_encode is None:
        # Rename columns to dimension_1, dimension_2, ...
        emb_df.columns = [f'dimension_{col}' for col in emb_df.columns]
        return emb_df

    if verbosity > 0:
        print(f"Encoding the data feature '{column_to_encode}'...")
    if explode_dim:
        # Rename columns to enc_dim_1, enc_dim_2, ...
        emb_df.columns = [f'enc_dim_{col}' for col in emb_df.columns]
        # Merge the embeddings with the data
        encoded_data_df = data.merge(emb_df, left_on=column_to_encode, right_index=True)
        # Drop the node column
        encoded_data_df.drop(column_to_encode, axis=1, inplace=True)
    else:
        encoded_data_df = data.copy()
        encoded_data_df[column_to_encode] = encoded_data_df[column_to_encode].apply(
            lambda x: model.kv.get_vector(str(x)))

    return encoded_data_df, model


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
