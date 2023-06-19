""" This module contains the functions to encode the datasets into a graph and to embed the graph. """
import networkx as nx
from pathlib import Path
from typing import Union


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
