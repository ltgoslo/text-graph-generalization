from graph_sampler import preprocess_graph
import json
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from tqdm import tqdm
import networkx as nx
import numpy as np
import pathlib
import rdflib


if __name__ == '__main__':
    raw_graphs = pathlib.Path("./raw_graphs/")
    n_nodes = []
    n_edges = []

    for input_file in tqdm(raw_graphs.iterdir()):
        rdf_graph = rdflib.Graph()
        rdf_graph.parse(input_file)
        original_graph: nx.MultiDiGraph = rdflib_to_networkx_multidigraph(rdf_graph)
        filtered_graph, possible_relations = preprocess_graph(original_graph)
        n_nodes.append(len(filtered_graph.nodes()))
        n_edges.append(len(filtered_graph.edges()))
    print(f"Mean nodes: {np.mean(n_nodes)}, std: {np.std(n_nodes)}")
    print(f"Mean edges: {np.mean(n_edges)}, std: {np.std(n_edges)}")

    stats = {"nodes":[np.mean(n_nodes), np.std(n_nodes)], "edges": [np.mean(n_edges), np.std(n_edges)]}
    with open("./graph_stats.json", "w") as f:
        json.dump(stats, f)
