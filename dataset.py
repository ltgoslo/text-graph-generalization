from device import device
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from utils import load_graph_from_path
import json
import networkx as nx
import torch
PLM = 'bert-base-uncased'


class TextGraphDataset(Dataset):
    def __init__(self, data, args, head_emb=None, tail_emb=None, tokenizer=None):
        self.special_token = "[SPECIAL_TOKEN]"
        self.special_node = "SPECIAL_NODE]"
        self.questions: List[str] = []
        self.question_paths: List[List[str]] = []
        self.paths: List[List[List[str], List[str]]] = []
        self.labels: List[int] = []
        self.sample_ids: List[str] = []
        self.graphs: List[Data] = []
        self.indicies: List[List[int]] = []
        self.unidirectional: bool = args.unidirectional
        self.bidirectional: bool = args.bidirectional
        self.model = AutoModel.from_pretrained(PLM, return_dict=True, output_hidden_states=True).to(device)
        node_embedding_size: int = self.model.config.hidden_size

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(PLM, use_fast=False)
            if args.bidirectional:
                assert args.unidirectional, "Unidirectional is set to false!"
                self.tokenizer.add_tokens(self.special_token, special_tokens=True)
                self.special_token_id = len(self.tokenizer) - 1
        else:
            self.tokenizer = tokenizer

        self.texts: Dict[str, List[int]] = {
            "input_ids": [],
            "attention_mask": []
        }

        if self.bidirectional:
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.unidirectional:
            self.special_node_indicies: List[List[int, int]] = []

        if head_emb is None and tail_emb is None:
            self.head_emb: torch.Tensor = torch.randn(node_embedding_size)
            self.tail_emb: torch.Tensor = torch.randn(node_embedding_size)
        else:
            self.head_emb: torch.Tensor = head_emb
            self.tail_emb: torch.Tensor = tail_emb

        with open("static/edge_type_dict.json", 'r') as f:
            e2i = json.load(f)

        for i in trange(len(data)):
            graph_path: str = data[i]["graph"]
            raw_graph: nx.MultiDiGraph = load_graph_from_path(graph_path)
            question: str = data[i]["question"]
            self.labels.append(data[i]["label"])
            self.sample_ids.append(i)
            self.questions.append(question)
            self.question_paths.append(data[i]["question_path"])
            self.paths.append(data[i]["og_path"])

            if self.bidirectional:
                question += self.special_token

            tokenized_question = self.tokenizer(question, padding="max_length", max_length=64)
            question_input_ids = torch.tensor(tokenized_question["input_ids"]).view(1, -1).to(device)
            question_attention_mask = torch.tensor(tokenized_question["attention_mask"]).view(1, -1).to(device)
            self.texts["input_ids"].append(question_input_ids[0])
            self.texts["attention_mask"].append(question_attention_mask[0])

            head_string = data[i]["og_path"][0][0]
            tail_string = data[i]["og_path"][-1][-1]
            if args.GNN_only or args.static:
                head = self.head_emb.tolist()
                tail = self.tail_emb.tolist()
            else:
                head_tokenized = self.tokenizer(question, head_string, padding="max_length", max_length=64)
                tail_tokenized = self.tokenizer(question, tail_string, padding="max_length", max_length=64)
                head_input_ids = torch.tensor(head_tokenized["input_ids"]).view(1, -1).to(device)
                tail_input_ids = torch.tensor(tail_tokenized["input_ids"]).view(1, -1).to(device)
                head_attention_mask = torch.tensor(head_tokenized["attention_mask"]).view(1, -1).to(device)
                tail_attention_mask = torch.tensor(tail_tokenized["attention_mask"]).view(1, -1).to(device)
                with torch.no_grad():
                    head_output = self.model(head_input_ids, head_attention_mask)[1].squeeze(0)
                    tail_output = self.model(tail_input_ids, tail_attention_mask)[1].squeeze(0)
                head = head_output.tolist()
                tail = tail_output.tolist()

                if self.unidirectional or self.bidirectional:
                    with torch.no_grad():
                        question_embedded = self.model(question_input_ids, question_attention_mask)[1].squeeze(0)
                    raw_graph.add_node(self.special_node)
                    for node in raw_graph.nodes():
                        if node == self.special_node:
                            continue
                        raw_graph.add_edge(self.special_node, node, type="special", id="special")
                        raw_graph.add_edge(node, self.special_node, type="special_r", id="special_r")

            edge_type_tensor = torch.tensor([e2i[e[-1]["type"]] for e in raw_graph.edges(data=True)], dtype=torch.long)
            adj_matrix = nx.adjacency_matrix(raw_graph)
            edge_index = from_scipy_sparse_matrix(adj_matrix)
            node_embeddings = {n: torch.zeros(node_embedding_size).tolist() for i, n in enumerate(list(raw_graph.nodes()))}
            node_embeddings[head_string] = head
            node_embeddings[tail_string] = tail
            node_keys = list(node_embeddings.keys())
            head_index = node_keys.index(head_string)
            tail_index = node_keys.index(tail_string)
            if self.unidirectional or self.bidirectional:
                node_embeddings[self.special_node] = question_embedded
                special_node_index = node_keys.index(self.special_node)
                self.special_node_indicies.append(special_node_index)
            nodes = torch.tensor(list(node_embeddings.values()), dtype=torch.float)
            processed_graph = Data(x=nodes, edge_index=edge_index[0], edge_type=edge_type_tensor).to(device)
            self.graphs.append(processed_graph)
            self.indicies.append([head_index, tail_index])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.texts["input_ids"][idx]
        attention_mask = self.texts["attention_mask"][idx]
        label = self.labels[idx]

        if self.unidirectional or self.bidirectional:
            special_node_index = self.special_node_indicies
        else:
            special_node_index = [0 for _ in self.indicies]

        if self.bidirectional:
            token = self.tokenizer(self.special_token, add_special_tokens=False)["input_ids"][0]
            special_token_index = input_ids.tolist().index(token)
        else:
            special_token_index = [0 for _ in self.indicies]

        return self.sample_ids[idx], torch.tensor(input_ids), torch.tensor(attention_mask), self.graphs[idx], self.indicies[idx], torch.tensor(special_node_index[idx]), torch.tensor(special_token_index), torch.tensor(label, dtype=torch.float32)

    def decode(self, idx):
        return self.tokenizer.decode(self.texts["input_ids"][idx], skip_special_tokens=False), self.labels[idx]
