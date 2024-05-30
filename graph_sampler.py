from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from rdflib.term import Literal
from typing import Any
import argparse
import collections
import json
import networkx as nx
import numpy
import pathlib
import random
import rdflib
import re
import sys
import tqdm


ENTITY_NAME_REGEX = re.compile(r'([A-Za-z]+)[0-9]+')


def get_entity_type(entity_name: str) -> str:
    """
    Get the type of an entity.

    This transforms "UndergraduateStudent143" into "UndergraduateStudent".
    """
    return ENTITY_NAME_REGEX.fullmatch(entity_name)[1]


def parse_args() -> argparse.Namespace:
    """ Return namespace containing command line arguments. """
    parser = argparse.ArgumentParser(description="Compositional reasoning over KGs and text using LM+GNNs: graph sampler.")
    parser.add_argument("--graph-output-path", type=pathlib.Path, default=None, help="Graphs output folder path.")
    parser.add_argument("--question-output-path", type=pathlib.Path, default=pathlib.Path("questions"), help="Questions output folder path.")
    parser.add_argument("--input-path", type=pathlib.Path, default=pathlib.Path("raw_graphs"), help="Location of the input graphs generated by LUBM.")
    parser.add_argument("--output-file-name", type=str, default="qs", help="Question file name.")
    parser.add_argument("--entities-distribution", type=str, default="uniform", help="How to sample points for positive and negative (one of: vertex, degree, vertex-then-degree, walks).")
    parser.add_argument("--entities-selection", type=str, default="unconditional", help="How to select the tail entity for positive and negative. Either unconditional or path.")
    parser.add_argument("-k", "--hops", type=int, default=3, help="The number of hops in the generated paths.")
    parser.add_argument("-n", "--samples", type=int, default=3, help="The number of samples to generate.")
    return parser.parse_args()


def get_entities_by_type(KG: nx.MultiDiGraph) -> dict[str, list[str]]:
    """ Get all vertices in a graph grouped by type. """
    entities_by_type: dict[str, set[str]] = collections.defaultdict(set)
    for entity in KG.nodes():
        entities_by_type[get_entity_type(entity)].add(entity)
    return {type: list(entities) for type, entities in entities_by_type.items()}


def get_degree_weighted_entities_by_type(KG: nx.MultiDiGraph) -> dict[str, list[str]]:
    """ For each entity type, get a list of vertices of this type where each vertex appears as many time as its degree. """
    entities_by_type: dict[str, list[str]] = collections.defaultdict(list)
    for edge in KG.edges():
        entities_by_type[get_entity_type(edge[0])].append(edge[0])
    return entities_by_type


def preprocess_graph(source: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, set[str]]:
    """
    Symmetrize graph, filter irrelevant vertices and relations.

    Literal entities such as phone numbers and RDF schema metadata are filtered out.
    All vertex and relation labels are converted to standard strings.
    For all arcs u->v labeled REL, a new arc v->u is created with label REL_r.

    Returns:
        nx.MultiDiGraph: the filtered graph
        set[str]: the set of relations appearing in the graph
    """
    relations: set[str] = set()
    KG = nx.MultiDiGraph()
    for i, (rdf_head, rdf_tail, rdf_pred) in enumerate(source.edges):
        if isinstance(rdf_head, Literal) or isinstance(rdf_tail, Literal):
            continue
        head: str = str(rdf_head).split('/')[-1]
        tail: str = str(rdf_tail).split('/')[-1]
        pred: str = str(rdf_pred).split('#')[-1]
        if 'type' in pred or 'import' in pred:
            continue
        if 'www' in head or 'www' in tail:
            continue
        pred_r: str = pred + '_r'
        KG.add_edge(head, tail, type=pred, key=pred)
        KG.add_edge(tail, head, type=pred_r, key=pred_r)
        relations.add(pred_r)
        relations.add(pred)
    return KG, relations


def get_spelled_out_path(KG: Any, list_of_vertices: list[str], only_relations: bool) -> list[Any]:
    """ Transform a list of vertices into a list of (head, relation, tail) or a list relations. """
    path: list[str | list[str]] = []
    for i in range(len(list_of_vertices)-1):
        head: str = list_of_vertices[i]
        tail: str = list_of_vertices[i+1]
        kg_edge: dict[str, Any] = KG.get_edge_data(head, tail)
        predicate: str
        if isinstance(KG, nx.MultiDiGraph):
            # In our "pruned" version of LUBM, there are no parallel edges going in the same direction.
            predicate = list(kg_edge.keys())[0]
        else: # nx.DiGraph
            predicate = kg_edge["type"]
        path.append(predicate if only_relations else [head, predicate, tail])
    return path


def select_probabilities(weights: dict[str, float], vertices: list[str]) -> numpy.ndarray:
    """ Given the weight of all vertices, construct the probability vector associated with a given subset of vertices. """
    output: numpy.ndarray = numpy.empty(len(vertices), float)
    for i, vertex in enumerate(vertices):
        output[i] = weights[vertex]
    return output / output.sum()


def sample_weighted_entities(KG: nx.MultiDiGraph, entities_by_type: dict[str, list[str]], hops: int, weights: list[dict[str, float]]) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Sample 4 vertices from a categorical distribution, making sure they can work as positives and negatives.

    Returns: (start1, end1), (start2, end2)
    The points satisfy the following properties:
        - start1 ≠ start2 or end1 ≠ end2
        - type(start1) = type(start2) and type(end1) = type(end2)
    """
    vertices: list[str] = list(KG.nodes())
    start1: str
    end1: str
    start1, end1 = numpy.random.choice(vertices, size=2, replace=False, p=select_probabilities(weights[0], vertices))
    start_type: str = get_entity_type(start1)
    end_type: str = get_entity_type(end1)

    start2: str = numpy.random.choice(entities_by_type[start_type], p=select_probabilities(weights[0], entities_by_type[start_type]))
    end2: str
    if start1 == start2:
        # Make sure end1 ≠ end2
        end2_weights: numpy.ndarray = numpy.array([0 if vertex == end1 else weights[0][vertex] for vertex in entities_by_type[end_type]], dtype=float)
        end2_weights /= end2_weights.sum()
        end2 = numpy.random.choice(entities_by_type[end_type], p=end2_weights)
    else:
        end2 = numpy.random.choice(entities_by_type[end_type], p=select_probabilities(weights[0], entities_by_type[end_type]))

    return (start1, end1), (start2, end2)


def sample_paths_endpoints(KG: nx.MultiDiGraph, entities_by_type: dict[str, list[str]], hops: int, weights: list[dict[str, float]]) -> tuple[tuple[str, str], tuple[str, str]]:
    """ Sample two paths and returns the endpoints, making sure these endpoints can work as positives and negatives. """
    def sample_path_endpoints(start_set: list[str]) -> tuple[None | str, None | str]:
        path: list[str] = [numpy.random.choice(start_set, p=select_probabilities(weights[0], start_set))]

        for hop in range(hops):
            previous_vertex: str = path[-1]
            candidates: list[str] = [vertex for vertex in KG.successors(previous_vertex) if vertex not in path]
            if not candidates:
                return None, None
            next_vertex: str = numpy.random.choice(candidates, p=select_probabilities(weights[hop+1], candidates))
            path.append(next_vertex)

        return path[0], path[-1]

    vertices: list[str] = list(KG.nodes())
    while True:
        start1, end1 = sample_path_endpoints(vertices)
        if start1 is not None:
            start2, end2 = sample_path_endpoints(entities_by_type[get_entity_type(start1)])
            if start2 is not None and start1 != start2 or start2 != end2:
                return (start1, end1), (start2, end2)


def compute_uniform_weights(KG: nx.MultiDiGraph, hops: int) -> list[dict[str, float]]:
    """ Make all vertices equally likely. """
    return [{vertex: 1 for vertex in KG.nodes()}] * (hops + 1)


def compute_degree_weights(KG: nx.MultiDiGraph, hops: int) -> list[dict[str, float]]:
    """ Select vertices according to their degree. """
    return [{vertex: KG.degree(vertex) for vertex in KG.nodes()}] * (hops + 1)


def compute_degree_then_uniform_weights(KG: nx.MultiDiGraph, hops: int) -> list[dict[str, float]]:
    """ Start selecting vertices according to their degree, then select neighbors uniformly. """
    return [{vertex: KG.degree(vertex) for vertex in KG.nodes()}] + [{vertex: 1 for vertex in KG.nodes()}] * hops


def compute_walks_weights(KG: nx.MultiDiGraph, hops: int) -> list[dict[str, float]]:
    """ Select vertices according to the number of walks starting there. """
    vertices: list[str] = list(KG.nodes())
    adj: numpy.ndarray = nx.to_numpy_array(KG, nodelist=vertices)
    walks: list[numpy.ndarray] = [numpy.ones(adj.shape[0])]
    for hop in range(hops):
        walks.append(walks[-1] @ adj)
    return [dict(zip(vertices, x)) for x in walks[::-1]]


def entities_sampler(args: argparse.Namespace, KG: nx.MultiDiGraph, entities_by_type: dict[str, list[str]], hops: int) -> tuple[tuple[str, str], tuple[str, str]]:
    """ Sample positive and negative entities. """
    weights: list[dict[str, float]]
    if args.entities_distribution == "uniform":
        weights = compute_uniform_weights(KG, hops)
    elif args.entities_distribution == "degree":
        weights = compute_degree_weights(KG, hops)
    elif args.entities_distribution == "degree-then-uniform":
        weights = compute_degree_then_uniform_weights(KG, hops)
    elif args.entities_distribution == "walks":
        weights = compute_walks_weights(KG, hops)
    else:
        print(f"Unknown entities-distribution: {args.entities_distribution}", file=sys.stderr)
        sys.exit(1)

    if args.entities_selection == "unconditional":
        return sample_weighted_entities(KG, entities_by_type, hops, weights)
    elif args.entities_selection == "path":
        return sample_paths_endpoints(KG, entities_by_type, hops, weights)
    else:
        print(f"Unknown entities-selection: {args.entities_selection}", file=sys.stderr)
        sys.exit(1)


def all_path_types(KG: nx.MultiDiGraph, start: str, end: str, hops: int) -> set[tuple[str, ...]]:
    """ Get the set of all path types between two given vertices. """
    alternative_paths: list[list[str]] = [n for n in nx.all_simple_paths(KG, start, end, hops+1) if len(n) == hops+1]
    rel_types_path: set[tuple[str, ...]] = set()
    for alternative in alternative_paths:
        relation_path: list[str] = get_spelled_out_path(KG, alternative, only_relations=True)
        rel_types_path.add(tuple(relation_path))
    return rel_types_path


def sample_question(args: argparse.Namespace, KG: nx.MultiDiGraph, entities_by_type: dict[str, list[str]]) -> None | tuple[tuple[str, ...], tuple[str, str], tuple[str, str]]:
    """ Sample a question with positive and negative pairs of entities. """
    while True:
        (start1, end1), (start2, end2) = entities_sampler(args, KG, entities_by_type, args.hops)
        path_types1: set[tuple[str, ...]] = all_path_types(KG, start1, end1, args.hops)
        path_types2: set[tuple[str, ...]] = all_path_types(KG, start2, end2, args.hops)

        type1_exclusives: set[tuple[str, ...]] = path_types1.difference(path_types2)
        type2_exclusives: set[tuple[str, ...]] = path_types2.difference(path_types1)
        # Make sure positives and negatives follow the same distribution
        if type1_exclusives and type2_exclusives:
            question: tuple[str, ...] = random.choice(list(type1_exclusives))
            return question, (start1, end1), (start2, end2)


def create_permutation(KG: nx.MultiDiGraph, entities_by_type: dict[str, list[str]]) -> dict[str, str]:
    """ Create a type-preserving permutation of vertex labels. """
    mapping: dict[str, str] = {}
    for entity_type, possibilities in entities_by_type.items():
        options: list[str] = possibilities
        random.shuffle(options)
        mapping.update(zip(possibilities, options))
    return mapping


if __name__ == '__main__':
    args = parse_args()

    # Create output directories if they do not exist
    for output_directory in [args.graph_output_path, args.question_output_path]:
        if not output_directory.is_dir():
            output_directory.mkdir(parents=True, exist_ok=True)

    question_path: pathlib.Path = args.question_output_path / f'{args.hops}_hop_{args.output_file_name}_{args.samples}.jsonl'
    with question_path.open("w") as question_file:
        for _, input_file in zip(tqdm.trange(args.samples//2), args.input_path.iterdir()):
            graph_path: pathlib.Path = args.graph_output_path / f'{input_file.stem}.graphml'

            rdf_graph = rdflib.Graph()
            rdf_graph.parse(input_file)
            original_graph: nx.MultiDiGraph = rdflib_to_networkx_multidigraph(rdf_graph)
            filtered_graph, possible_relations = preprocess_graph(original_graph)
            entities_by_type: dict[str, list[str]] = get_entities_by_type(filtered_graph)

            # Permute node labels to remove noise from the LUBM generator (e.g Professor1 always writes Paper1 etc..).
            mapping: dict[str, str] = create_permutation(filtered_graph, entities_by_type)
            KG: nx.MultiDiGraph = nx.relabel_nodes(filtered_graph, mapping)

            path, pos, neg = sample_question(args, KG, entities_by_type)
            start_node_pos, end_node_pos = pos
            start_node_neg, end_node_neg = neg

            def make_entry(polarity: bool, start_node: str, end_node: str) -> dict[str, Any]:
                return {
                    'question': 'PLACEHOLDER',
                    'question_path': path,
                    'label': polarity,
                    'graph': str(graph_path),
                    'og_path': [[start_node], [end_node]],
                    'k': args.hops
                }

            json.dump(make_entry(True, start_node_pos, end_node_pos), question_file)
            question_file.write("\n")
            json.dump(make_entry(False, start_node_neg, end_node_neg), question_file)
            question_file.write("\n")
            nx.write_graphml_lxml(KG, graph_path)