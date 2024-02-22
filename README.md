# The Loop Community Algorithm

The Loop Community Algorithm is designed to identify communities in dense, directed networks with heterogeneous weights.

## Introduction

Welcome, everyone! Thank you for taking the time to explore this software.

Community detection is a significant challenge as it is closely linked to data encoding. With vast amounts of data generated daily worldwide, the ability to process and uncover its underlying structures, symmetries, and global features is essential. Finding communities within data is akin to breaking it down into more manageable and homogeneous segments, facilitating faster interpretation, similar to neatly folding clothes for easier storage. In real-world networks, communities naturally emerge as a way to minimize entropy, making information retrieval or specific actions more efficient, a crucial property under natural selection pressures.

While numerous algorithms exist for community detection in various network types, finding communities in dense, directed networks with heterogeneous weight distributions, such as the macaque retrograde tract-tracing network, remains an open question. Therefore, we have developed an algorithm to address the challenges of identifying communities in such complex networks.

Our journey led us to the link community algorithm, which has significant features like assigning nodes to multiple clusters. However, we enhanced the algorithm to meet the requirements of anatomical neural networks, resulting in our algorithm named. This modified version extends the algorithm's applicability to cortical networks and, more generally, to (un)directed, sparse/dense, (un)weighted, and simple networks (without self-loops and multiple links between the same nodes).

## New Features

To adapt the link community algorithm for broader network types, we introduced several key enhancements:

**Novel Link Neighborhood Concepts**: We defined novel link neighborhood concepts for directed graphs, diverging from existing literature interpretations, especially regarding line digraphs. These new criteria have significant graph-theoretical implications yet to be fully explored.

**Node Community Dendrogram**: We developed an algorithm that projects the link merging process into a node merging process, allowing the generation of a node community dendrogram. This dendrogram simplifies the search for meaningful clusters, preserving the original intention of partitioning nodes into several groups or **covers**.

**Computing Similarities Using Different Topologies**: Our algorithm allows users to determine node communities based on the similarity of source and/or target connections. This feature is particularly useful in directed networks, where nodes may have different functional roles based on their interaction directions.

**Novel Quality Function for Link Communities (Loop Entropy)**: Introducing the loop entropy ($S_{L}$) as a quality function addresses the challenge of dense networks where modularity may not be effective. Loop entropy measures the quality of a link partition based on the amount of informative link community information, providing insights into the network's structure.

## Unique Attributes of Our Algorithm

- **Focus on Anatomical Neural Networks**: Our algorithm specializes in finding the community structure of the dense, directed, and heterogeneous macaque retrograde tract-tracing network, a challenging domain due to experimental limitations and network complexity.

- **Node Hierarchy Extraction**: Unlike traditional community algorithms, our algorithm can extract a node hierarchy from network similarity information. This hierarchy enables the exploration of different community scales within the network, aiding interpretation.

- **Ability to Handle Dense Networks**: Our algorithm's loop entropy approach enables effective partitioning in dense networks, addressing a common challenge in community detection.

## Pybind11 C++ Libraries

The algorithm is primarily implemented in Python (3.9.13), with C++ libraries used to enhance processing speed. The integration of C++ code is facilitated using pybind11.

## Examples

We provide several examples in Jupyter Notebooks to demonstrate the algorithm's usage:

- **ER_example**: Illustrates the algorithm's behavior in an Erdos-Renyi random graph with high density.
- **HRG_example**: Explores the algorithm's performance in a sparse directed hierarchical random graph.
- **BowTie_examples**: Demonstrates the algorithm's capabilities in networks with topological Network of Communities (NOC) structures.
- **HSF_example**: Shows how the algorithm handles hierarchical scale-free networks.

## Open Questions

While our algorithm has demonstrated effectiveness, there are areas for improvement, including computational speed and the refinement of cover assignment algorithms.

## References

- Ahn, Y. Y., Bagrow, J., & Lehmann, S. (2010). Link communities reveal multiscale network complexity. Nature, 466(7307), 761–764.
- Lancichinetti, A., & Fortunato, S. (2009). Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities. Phys. Rev. E, 80, 016118.
- Markov, N. T., et al. (2011). Weight consistency specifies the regularities of macaque cortical networks. Cereb. Cortex, 21(6), 1254–1272.
- Markov, N. T., et al. (2012). A weighted and directed interareal connectivity matrix for macaque cerebral cortex. Cereb. Cortex. doi:10.1093/cercor/bhs1270.
- Müllner, D. (2013). fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python. Journal of Statistical Software, 53(9), 1–18.