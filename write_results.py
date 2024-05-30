from typing import List, Dict
import json
import jsonlines
import numpy as np


def get_acc(objects) -> float:
    predictions: List[float] = []
    for obj in objects:
        predictions.append(float(obj["pred"] == obj["label"]))
    return np.mean(predictions)


def get_scores_normal(model_name: str, seeds: List[int] = [42, 101, 987, 64, 923]) -> Dict[str, float]:
    runs = []
    for seed in seeds:
        with jsonlines.open(f"./results/{model_name}_seed_{seed}.jsonl", 'r') as f:
            samples = [s for s in f]
            runs.append(samples)
    scores = [get_acc(r) for r in runs]
    return {"mean": round(np.mean(scores)*100, 2), "std": round(np.std(scores)*100, 2)}


def get_scores_system(model_name: str, hop) -> Dict[str, float]:
    runs = []
    print(model_name)
    for fold in [0, 1, 2, 3]:
        with jsonlines.open(f"./results/{model_name}_super_system_round_{rr}_hop_{hop}_fold_{fold}_seed_42.jsonl", 'r') as f:
            samples = [s for s in f]
            runs.append(samples)
    scores = [get_acc(r) for r in runs]
    print(scores)
    return {"mean": round(np.mean(scores)*100, 2), "std": round(np.std(scores)*100, 2)}


if __name__ == "__main__":
    print("Write results")
    models = ["disjoint", "grounded", "gnn", "unidirectional", "bidirectional"]
    tests = ["subst_2_hop", "subst_3_hop", "subst_4_hop", "prod_T2a3_V4", "prod_T2a4_V3", "prod_T3a4_V2"]
    results = {}
    for model in models:
        for test in tests:
            test_path = f"{model}_{test}"
            results[test_path] = get_scores_normal(test_path)
        for hop in [2, 3, 4]:
            results[f"{model}_system_{hop}_hop"] = get_scores_system(model, hop)
    print(results)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
