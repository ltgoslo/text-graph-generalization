<h2 align="center"><b>Compositional Generalization with Grounded Language Models</h2><br></b>

<p align="center">
  <b>Sondre Wold, Étienne Simon, Lucas Georges Gabriel Charpentier, Egor V. Kostylev, Erik Velldal, Lilja Øvrelid</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
  </i>
  <br>
</p>
<br>
<p align="center">
  <a href=""><b>Paper</b></a><br>
</p>

_______

### Downloding and preprocessing the data

Details on how to generate the raw LUBM graphs can be found in
`download_data.md`

### Sampling

Sampling the pairs of entities in the graphs that are later translated into
the questions is done through `graph_sampler.py`.

The sampler takes as input the raw graphs produced by the LUBM UBA and
outputs two things: a folder containing graph objects in `graphml` format,
and a folder that has a `jsonl` file that contains all the dataset samples
for the GNN encoder. 

E.g, in order to produce all the dataset samples for 2 hops, you can run the
following:

```
python graph_sampler.py
    --graph-output-path data/2_hop_graphs/
    --question-output-path data/2_hop_questions/
    --input-path raw_graphs
    -k 2
    -n 30000
```

To produce the dataset samples for 3 and 4 hops, run the same command but
swap the folder names and the `k` argument accordingly.

### Creating the dataset splits
Run `dataset.py` with `--task` being either `sys`, `subst` or `prod` to
create the datasets used for the three experiments from the paper.

### Running the experiments
`train.py` runs all the experiments on the provided dataset splits. An axplantion of the CLI can be found in the `argpase` method in this file. The file runs the experiments over a set of seeds
provided by a white space seperated list to `--seed`.  

The different models are selected by setting the following model-specific
flags:
- Baseline (gnn-only): `--gnn`
- Disjoint: `--static`
- Grounded: no flag
- Unidirectional: `--unidirectional`
- Bidirectional: `--unidirectional --bidirectional`

The hyperparameters we used for the experiments in the paper can be found
in Appendix B.

### Requirements

The experiments ran using the following specific packages, with Python
`3.9.12`:

```
jsonlines==4.0.0
networkx==3.2.1
numpy==1.26.3
safetensors==0.4.2
tokenizers==0.15.1
torch==1.11.0+rocm4.5.2
torch-cluster==1.6.3
torch-scatter==2.1.2
torch-sparse==0.6.18
torch-spline-conv==1.2.2
torch_geometric==2.4.0
transformers==4.37.1
```

### Results
Results are logged to Wandb and written to a json file.  The `--debug`
flags disables Wandb logging.
You can use ```write_results.py``` to aggreate the results across seeds and
all models. 
