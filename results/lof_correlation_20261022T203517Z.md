# Embedding-based anomaly score vs. real GZ2 volunteer labels

Sample size: 300 galaxies, randomly drawn (not pre-filtered).

Method: Local Outlier Factor (n_neighbors=20) on raw model embeddings.

Higher LOF score = more of an outlier in that model's embedding space.

Compared against Class6.1 = the real GZ2 volunteer vote fraction for "is there anything odd?".


| model | Spearman r (LOF score vs Class6.1) | p-value |
|---|---|---|
| zoobot | 0.403 | 0.0000 |
| clip | 0.126 | 0.0296 |
