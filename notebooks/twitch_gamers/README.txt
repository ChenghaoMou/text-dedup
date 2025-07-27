About the dataset:

A social network of Twitch users which was collected from the public API in Spring 2018. Nodes are Twitch users and edges are mutual follower relationships between them. The graph forms a single strongly connected component without missing attributes. The machine learning tasks related to the graph are count data regression and node classification. There are 6 specific tasks:

- Explicit content streamer identification.
- Broadcaster language prediction.
- User lifetime estimation.
- Churn prediction.
- Affiliate status identification.
- View count estimation.

Statistics:

Nodes 168,114
Edges 6,797,557
Density 0.0005
Transitivity 0.0184

Citing:

@misc{rozemberczki2021twitch,
      title={Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings},
      author={Benedek Rozemberczki and Rik Sarkar},
      year={2021},
      eprint={2101.03091},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}

Twitch Gamers paper:

https://arxiv.org/abs/2005.07959

Twitch Gamers project:

https://github.com/benedekrozemberczki/datasets
