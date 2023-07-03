"""
Query OpenML for the datasets' meta-features.

Example usage:

from node2vec import Node2Vec

from pathlib import Path
from pyvis.network import Network
from src.load_datasets import load_dataset


DATA_DIR = Path("./data")
df = load_dataset(DATA_DIR / "dataset.csv")

# ---- get the meta-features
meta_features = get_metafeatures(df.dataset.unique().astype(str))
# ... You can then build an encoder from meta_features (a left join should do the trick)

# ---- get the encoder graph
G = load_graph(DATA_DIR / "encoders.adjlist")

# -- embed the graph
n2v = Node2Vec(G, dimensions=2, walk_length=20, num_walks=1000, workers=1, quiet=True)
embedding = n2v.fit().wv.vectors
# ...  You can then build an encoder with the embedding
# ... If you embed in 2d, you can plot an annotatate scatter plot of the embedding to see if it worked as expected

# -- visualize the graph
nt = Network('1000px', '1000px')
nt.from_nx(G)
nt.show('encoder_graph.html', notebook=False)  # it will store the html file
"""

import networkx as nx
import pandas as pd

from openml.datasets import get_datasets
from pathlib import Path
from typing import List, Union


def get_metafeatures(datasets: List[Union[str, int]]) -> pd.DataFrame:
    return pd.DataFrame([d.qualities.update({"id": d.id}) or d.qualities
                         for d in get_datasets(datasets)]).set_index("id")


def load_graph(path: Union[Path, str]) -> nx.Graph:
    return nx.read_adjlist(path)


class NoY(object):
    """
    As category_encoders.OneHotEncoder does not support multioutput, we need to fool it by just not passing Y to it.
    Alternatively, just pre-process the dataset separately.
    """

    def __init__(self, encoder):
        self.encoder = encoder

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)

    def fit_transform(self, X, y=None):
        return self.encoder.fit_transform(X)