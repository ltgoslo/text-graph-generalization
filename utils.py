import networkx as nx
import torch
import numpy as np
import random

def load_graph_from_path(path: str) -> nx.Graph:
    '''
    Load a graph from a given path.

    Args:
        path (str): The path to the graph file.

    Returns:
        G (networkx.Graph): The loaded graph.

    '''
    G = nx.read_graphml(f'{path}')
    return G

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True