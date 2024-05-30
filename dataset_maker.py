from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
import argparse
import copy
import json
import jsonlines
import logging
import math
import networkx as nx
import random


POSSIBLE_TASKS = ["sub", "pro", "sys"]
SUB_TRAINING_SIZE = 10000
SUB_TEST_SIZE = 2000
ANNOTAITON_MAP_PATH = Path("./static/annotation_map.json")
HOP_2_ANNOTATIONS = Path("./static/2_hop_annotations.json")
HOP_3_ANNOTATIONS = Path("./static/3_hop_annotations.json")
HOP_4_ANNOTATIONS = Path("./static/4_hop_annotations.json")
HOP_2_PATH = Path("./data/2_hop_questions/2_hop_qs_25000.jsonl")
HOP_3_PATH = Path("./data/3_hop_questions/3_hop_qs_25000.jsonl")
HOP_4_PATH = Path("./data/4_hop_questions/4_hop_qs_25000.jsonl")


def chunking(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sub", help=f"Which task to produce dataset splits for (one of: {' '.join(POSSIBLE_TASKS)}).")
    parser.add_argument("--hop_2_path", type=Path, default=HOP_2_PATH)
    parser.add_argument("--hop_3_path", type=Path, default=HOP_3_PATH)
    parser.add_argument("--hop_4_path", type=Path, default=HOP_4_PATH)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--round", type=int, help="what round to execute for the sys task")
    return parser.parse_args()


def get_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        amap = json.load(f)
    return amap


def get_jsonl(path: Path) -> List:
    with jsonlines.open(path, 'r') as f:
        lines = [line for line in f]
    return lines


def write_jsonl(path: Path, data: List) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(path, 'w') as f:
        f.write_all(data)
    logging.info(f"Wrote {path} to disk...")


def get_annotation_map(path: Path):
    index_2_relation = get_json(path)
    relations_2_id = {}
    for relation, value in index_2_relation.items():
        relations_2_id[tuple(value['type'])] = relation
    return index_2_relation, relations_2_id


def create_individual_hop_files(hop_path: Path, index_2_relation: Dict, relations_2_id: Dict):
    raw_qs = get_jsonl(hop_path)
    questions = []
    errors = 0
    lacking_annotation = 0
    unseen_relation_types = set()
    for question in raw_qs:
        if question['og_path'][0][0] is None:
            errors += 1
            continue
        idx = relations_2_id.get(tuple(question['question_path']), -1)
        if idx == -1:
            unseen_relation_types.add(tuple(question['question_path']))
            lacking_annotation += 1
            continue
        annotation = index_2_relation[idx]['mapping']
        q = annotation.replace('X', question['og_path'][0][0])
        q = q.replace('Y', question['og_path'][-1][-1])
        question['question'] = q
        questions.append(question)
    return questions, errors, lacking_annotation, unseen_relation_types


def assert_primitives(train_relations: List, test_relations: List) -> bool:
    train_primitives = set()
    test_primitives = set()
    for r in train_relations:
        for t in r:
            train_primitives.add(t)
    for r in test_relations:
        for t in r:
            test_primitives.add(t)
    good_2_go = True
    for p in test_primitives:
        if p not in train_primitives:
            good_2_go = False
    return good_2_go


def load_graph_from_path(path: str) -> nx.Graph:
    G = nx.read_graphml(f'{path}')
    return G


def get_spelled_out_path(KG, list_of_vertices, only_relations):
    """ Transform a list of vertices into a list of (head, relation, tail) or a list relations. """
    path: list[str | list[str]] = []
    for i in range(len(list_of_vertices)-1):
        head: str = list_of_vertices[i]
        tail: str = list_of_vertices[i+1]
        kg_edge: dict[str, Any] = KG.get_edge_data(head, tail)
        predicate: str
        if isinstance(KG, nx.MultiDiGraph):
            predicate = list(kg_edge.keys())[0]  # no parallel edges going in the same direction.
        else:  # nx.DiGraph
            predicate = kg_edge["type"]
        path.append(predicate if only_relations else [head, predicate, tail])
    return path


def prune_data(data_points: List[List], test_relations: List, hop: int) -> List:
    pruned_samples = []
    for point in tqdm(data_points):
        graph_path = point["graph"]
        KG = load_graph_from_path(graph_path)
        head_p = point["og_path"][0][0]
        tail_p = point["og_path"][-1][-1]
        paths = [n for n in nx.all_simple_paths(KG, head_p, tail_p, hop+1) if len(n) == hop+1]
        to_break = False
        for path in paths:
            path_key = tuple(get_spelled_out_path(KG, path, True))
            if path_key in test_relations:
                to_break = True
                continue
        if not to_break:
            pruned_samples.append(point)
    return pruned_samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Dataset creator")
    args = parse_args()
    assert args.task in POSSIBLE_TASKS, exit(f"Invalid task argument provided, use one of: {' '.join(POSSIBLE_TASKS)}")
    index_2_relation, relations_2_id = get_annotation_map(ANNOTAITON_MAP_PATH)

    hop_2_questions, hop_2_errors, hop_2_lacking_annotation, hop_2_unseen_relation_types = create_individual_hop_files(args.hop_2_path, index_2_relation, relations_2_id)
    logging.info(f"Parsed 2 hops, total question: {len(hop_2_questions)}, errors: {hop_2_errors}, missing annotation: {hop_2_lacking_annotation}")

    hop_3_questions, hop_3_errors, hop_3_lacking_annotation, hop_3_unseen_relation_types = create_individual_hop_files(args.hop_3_path, index_2_relation, relations_2_id)
    logging.info(f"Parsed 3 hops, total question: {len(hop_3_questions)}, errors: {hop_3_errors}, missing annotation: {hop_3_lacking_annotation}")

    hop_4_questions, hop_4_errors, hop_4_lacking_annotation, hop_4_unseen_relation_types = create_individual_hop_files(args.hop_4_path, index_2_relation, relations_2_id)
    logging.info(f"Parsed 4 hops, total question: {len(hop_4_questions)}, errors: {hop_4_errors}, missing annotation: {hop_4_lacking_annotation}")

    # Creating dataset splits
    if args.task == "sub":
        save_path: Path = Path("./data/substitutivity")
        for hop, qs in zip(["2", "3", "4"], [hop_2_questions, hop_3_questions, hop_4_questions]):
            if args.shuffle:
                random.shuffle(qs)
            train_data = qs[:SUB_TRAINING_SIZE]
            test_data = qs[SUB_TRAINING_SIZE:SUB_TRAINING_SIZE+SUB_TEST_SIZE]
            train_path = save_path / f"{hop}_hop_train.jsonl"
            test_path = save_path / f"{hop}_hop_test.jsonl"
            write_jsonl(train_path, train_data)
            write_jsonl(test_path, test_data)
    elif args.task == "pro":
        save_path: Path = Path("./data/productivity")
        if args.shuffle:
            random.shuffle(hop_2_questions)
            random.shuffle(hop_3_questions)
            random.shuffle(hop_4_questions)
        train_2 = hop_2_questions[:5000]
        train_3 = hop_3_questions[:5000]
        train_4 = hop_4_questions[:5000]
        test_2 = hop_2_questions[-2000:]
        test_3 = hop_3_questions[-2000:]
        test_4 = hop_4_questions[-2000:]
        train_2_and_3 = train_2 + train_3
        train_2_and_4 = train_2 + train_4
        train_3_and_4 = train_3 + train_4
        train_2_and_3_path = save_path / "2and3_train.jsonl"
        train_2_and_4_path = save_path / "2and4_train.jsonl"
        train_3_and_4_path = save_path / "3and4_train.jsonl"
        test_2_path = save_path / "2_test.jsonl"
        test_3_path = save_path / "3_test.jsonl"
        test_4_path = save_path / "4_test.jsonl"
        write_jsonl(train_2_and_3_path, train_2_and_3)
        write_jsonl(train_2_and_4_path, train_2_and_4)
        write_jsonl(train_3_and_4_path, train_3_and_4)
        write_jsonl(test_2_path, test_2)
        write_jsonl(test_3_path, test_3)
        write_jsonl(test_4_path, test_4)
    elif args.task == "sys":
        print(f"Creating data for round {i_round}")
        _, hop2_rel_to_id = get_annotation_map(HOP_2_ANNOTATIONS)
        _, hop3_rel_to_id = get_annotation_map(HOP_3_ANNOTATIONS)
        _, hop4_rel_to_id = get_annotation_map(HOP_4_ANNOTATIONS)
        save_path: Path = Path("./data/systematicity")
        hop_to_data = {
                2: hop_2_questions,
                3: hop_3_questions,
                4: hop_4_questions,
            }
        n_chunks = 5
        hop_to_percentage = {
                2: 0.3,
                3: 0.2,
                4: 0.1
            }
        for hop, hop_map in zip([2, 3, 4], [hop2_rel_to_id, hop3_rel_to_id, hop4_rel_to_id]):
            has_valid_setup = False
            tries = 10
            while not has_valid_setup:
                if tries == 0:
                    exit(f"Failed to create split for {hop}-hop setting")
                possible_relations = list(hop_map.keys())
                sample_size = math.ceil(hop_to_percentage[hop] * len(possible_relations))
                test_relations = random.sample(possible_relations, sample_size)
                train_relations = [r for r in possible_relations if r not in test_relations]
                has_valid_setup = assert_primitives(train_relations, test_relations)
                tries -= 1

            test_data = []
            tmp_train_data = []
            rouge_points = 0
            for data_point in hop_to_data[hop]:
                relation = tuple(data_point["question_path"])
                if relation in train_relations:
                    tmp_train_data.append(data_point)
                elif relation in test_relations:
                    test_data.append(data_point)
                else:
                    rouge_points += 1

            logging.info(f"Found rouge data points count: {rouge_points}")
            logging.info(f"Total train size: {len(tmp_train_data)}")
            logging.info(f"Total test size: {len(test_data)}")
            pruned_train_data = prune_data(tmp_train_data, test_relations, hop)
            logging.info(f"Pruned {1 - (len(pruned_train_data) / len(tmp_train_data))}% for {hop}-hop")
            chunks = list(chunking(pruned_train_data, len(pruned_train_data) // n_chunks))
            logging.info(f"N-chunks: {len(chunks)}")
            for i in range(0, len(chunks)-1):
                td = copy.deepcopy(chunks)
                popped = td.pop(i)
                td = [j for sub in td for j in sub]
                output_path = save_path / f"round_{i_round}_{hop}hop_fold{i}_train.jsonl"
                write_jsonl(output_path, td)
            test_output_path = save_path / f"round_{i_round}_{hop}_test.jsonl"
            write_jsonl(test_output_path, test_data)
