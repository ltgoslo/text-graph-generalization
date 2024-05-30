<h2 align="center"><b><Compositional Generalization with Grounded Language Models/h2><br></b>


<p align="center">
  <b>Sondre Wold, Étienne Simon, Lucas Georges Gabriel Charpentier, Egor V. Kostylev, Erik Velldal, Lilja Øvrelid</b>
</p>

<p align="center">
  <i>
    <sup>*</sup>University of Oslo<br>
  </i>
  <br>
</p>
<br>
Grounded language models use external sources of information, such as
knowledge graphs, to meet some of the general challenges associated with
pre-training. By extending previous work on compositional generalization in
semantic parsing, we allow for a controlled evaluation of the degree to
which grounded language models can learn and generalize from patterns in
knowledge graphs. We develop a procedure for generating natural language
questions paired with knowledge graphs that targets different aspects of
compositionality and further avoids grounding the language models in
information already encoded implicitly in their weights. We evaluate
existing methods for combining language models with knowledge graphs and
find that all of them struggle with generalization to sequences of unseen
lengths and to novel combinations of seen base components. While our
experimental results provide some insight into the expressive power of
these models, we hope our work and released datasets motivate future
research on how to better combine language models with structured knowledge
representations.
<p align="center">
  <a href="https://arxiv.org/abs/2404.01196"><b>Paper</b></a><br>
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
