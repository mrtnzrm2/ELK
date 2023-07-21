# ELK
ELK is a link community algorithm created to find communities in dense, directed networks with heterogeneous weights.

 ## Introduction
 Welcome everyone! How good of you to spend some time looking at this software. I consider this software an extension of the original link community algorithm ([Ahn et al., 2010](https://doi.org/10.1038/nature09182)).
 
 The problem of community detection is profound since it is connected to the problem of data encoding. Each day, millions of bytes of information are saved by governments and companies all around the world. However, all the data is only helpful if you can process it and reveal its structures, symmetries, global features, and laws. Finding communities is like separating the data into homogeneous pieces. In the context of information theory, partitioning the data into clusters can allow you to decode the information faster since the data is arranged correctly, like your clothes when you decide to fold them nicely, which is easier to interpret. Networks in the real world naturally form communities since it is the way to minimize entropy, steps to reach specific information or execute a particular action, a property crucial if the system is under the forces of natural selection.

 There are several community detection algorithms for different network classes. However, finding communities in **dense**, **directed**, and **heterogeneous weight** distribution networks, as the macaque retrograde tract-tracing network (Markov et al. 2011 and 2012), is still an open question since it is not clear what a community means in such complex system. Therefore, we started to work on an algorithm to overcome the challenges of identifying communities in this type of network.

Our journey led us to the link community algorithm, which has many essential features, such as assigning nodes to multiple clusters. Nevertheless, to make it work in anatomical neural networks, we had to add several features to the algorithm. We have baptized the algorithm **ELK** to distinguish it from the original. However, the new features allow the usage of the algorithm to cortical networks, and, in general, any (un)directed, sparse/dense, (un)weighted, and simple network (without self-loops and multiple links between the same nodes).

## New features
We had to add several pieces to improve the link community algorithm to use it in more general networks. The most important are the following:

1. **Novel link neighborhood concepts**: We introduce novel link neighborhood definitions for directed graphs. Our interpretation diverges from the literature, especially the definition of line digraphs. The new criteria have many graph-theoretical implications that are still undiscovered.

2. **Node community dendrogram**: Link communities, when introduced, were exciting since nodes naturally can belong to multiple groups through their links. This property is natural in many social, economic, and biological networks. However, interpreting link communities is not obvious, and associated node community structures are needed to understand the network's structure.

We made an algorithm that projects the link merging process to a node merging process, allowing one to obtain a node community dendrogram that significantly simplifies the search for meaningful clusters. This tactic solves the problem since a hierarchy is the data structure that allows nodes to belong to one group at a level or split up into several in another, preserving the original intention of link communities to partition the nodes into several groups or **covers**.

3. **Computing similarities using different topologies**: You can choose to find node communities for how similar their source or/and target connections are. In directed networks, nodes can have different functions from the perspective of acting or receiving the action of other nodes. Our algorithm can produce partitions considering the direction of interest, making them easier to interpret.

4. **Novel quality function for link communities: The loop entropy ($H_{L}$)**: As it is well known, the concept of a community can have multiple interpretations; however, it is well accepted that the communities tend to be formed by the set of nodes with more connections between them than with the rest of the network. But what happens when the network is so dense that modularity, i.e., the density of a cluster compared to a random network, stops being a good quality function to detect the best partition?

 To solve that problem, we introduce the loop entropy, $H_{L}$, which measures the quality of a link partition for the amount of useful link community information. In information theory, entropy measures the expected information of a random variable. In this case, the random variable is the distribution of effective links in a link community in the network. The number of effective links in a link community is the number of excess links respect to a tree version of that link community.

 In a tree network with $n$ nodes, there are $m=n-1$ links. Then, the number of effective links in a link community $c$ is

$m_{f}^{c} = m^{c} - (n^{c} - 1)$,

The total number of possible effective links in the nework is $M - N + 1$, where $M$ and $N$ are the total number of (un)directed links and nodes in the network. Then, the probability of picking at random an effective link from the link community $c$ is

$p_{c} = \frac{m_{f}^{c}}{M-N+1}$.

On the other hand, the probability of picking an uneffective link is

$q = 1 - \frac{\sum_{c} m_{f}^{c}}{M-N+1}$.

Then, the loop entropy of the link partition is

$H_{L} = -\sum_{c}p_{c}\log(p_{c}) -q\log(q)$.

By selecting the link partition with the highest loop entropy, we select the state the link partition with the highest useful link community information which, in the same time, is the state between the domination of small tree-like link communities and complex loop-like link communities.
 
 We have tested the loop entropy and averange link density quality functions to find the node partition of [LFR](https://doi.org/10.1103/PhysRevE.80.016118) benchmark networks. The results show that both quality functions work similarly, but the loop entropy still works in dense networks.

## Why is ELK different from the rest of the community detection algorithms?
- We target our attention on finding the community structure of the anatomical neural network of retrograde tract tracing experiments of the macaque monkey. The network is known to be dense, directed, heterogeneous, and, because of experimental challenges, only a subgraph from the total graph is known. The complete network hast 106 nodes and is denoted as $G_{106\times 106}$; however, we have a subset $G_{106\times 57}$ representing the inlinks and number of neuros to $57$ areas from the whole atlas. Our goal is to find communities in the edge-complete subgraph $G_{57\times 57}$ but considering the whole measured subgrpah $G_{106\times 57}$. One difference is that there are not many community detection algorithms that can find partitions in a subgraph using the information of the connections outside of the graph, as happens in our case.

- Another difference is that our algorithm can extract a node hierarchy from the similarity infomation of the network. Using the node hierarchy, we can study the hierarchical organization of the network. Traditionally, community algorithms try to find the community structure with largest modularity or likelihood. By finding a node hierarchy we can find other meaningful scales in the network. This property is inherated by the original link community algorithm; however, without the node hierarchy, it is challenging to interpret and find the different community scales of a network.

- A third difference is that our algorithm can work in dense networks since the most relevant link and node partition can be found using the loop entropy.


## Pybind11 C++ libraries
The code is implemented primarily in Python (3.9.13), with some C++ libraries used to speed up the algorithm.

The steps to pybind (mandatory) the C++ code are the following:

1. Install **cmake** version [3.24.0-rc4](https://cmake.org/files/). To use CMake, remember to add it to your path.

```
export PATH="/Applications/CMake.app/Contents/bin:/usr/local/bin:$PATH"
```

2. Install pybind11.

```
pip3 install pybind11
```

3. Download the hclust-cpp repository created by [Daniel Müllner](http://danifold.net/) and [Christoph Dalitz](https://lionel.kr.hs-niederrhein.de/~dalitz/data/hclust/).

```
https://github.com/cdalitz/hclust-cpp.git
```
4. Paste the repository in the cpp/process_hclust/src and cpp/la_arbre_a_merde/src.

5. Install the C++ libraries in python by running:

```
pip3 install cpp/simquest
pip3 install cpp/process_hclust
pip3 install cpp/la_arbre_a_merde
```

## Examples
We have created several examples in Jupyter Notebooks to understand better how to use the algorithm.

- ER_example: Running the algorithm in an Erdos-Renyi random graph with high density. The lack of structure in the node dendrogram shows that the algorithm does not find communities in this null model.

- HRG_example: Explore the algorithm's performance in a sparse directed hierarchical random graph. The most remarkable aspect is that even if the quality functions find different partitions, all the clustering information is encoded in the node community hierarchy. From there, one can read which nodes have a clear modular or overlapping role and the interregional distance between nodes at different hierarchy levels.

- BowTie_example_one: We will show how the algorithm works in a network with a topological NOC. A topological NOC appears because its links belong to two link communities which form two node communities. This situation will split the links into different link communities that sharply contrast with the monotonous membership of the links from nodes that only belong to one community.

- BowTie_example_two: In dense weighted networks, NOCs appear not only because of the lack of connections between groups of nodes but also for a contrasting weighted connectivity profile. The link community algorithm can identify this second type of NOC.

- HSF_example: There are hierarchical scale-free networks besides the traditional hierarchical graphs as denser communities inside sparser ones. This network combines the hierarchical structure with the presence of hubs and the lack of scale, i.e., each level of the hierarchy replicates the level below. We can find the clusters in this complex network using a binary similarity index and ELK.

## Open questions
There is still plenty of work to do. Some of the points to improve are:

- Low computational speed. Currently, the processing of link communities to identify the most exciting partitions is slow and scales as $O(M^{2})$ where $M$ is the number of links in the network. The link-to-node dendrogram projection also scales in the same way.

- The algorithm identifies the NOCs but the algorithm to assign covers to them is very simple. Although it has been proved to work in LFR benchmarks, further research is needed to understand better if the extention of the cover predictability.

## References
- Ahn, YY., Bagrow, J. & Lehmann, S. Link communities reveal multiscale network complexity. Nature 466, 761–764 (2010). https://doi.org/10.1038/nature09182
- Lancichinetti, A., & Fortunato, S. (2009). Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities. Phys. Rev. E, 80, 016118.
- Markov, N.T., Misery, P., Falchier, A., Lamy, C., Vezoli, J., Quilodran, R., Gariel,
M.A., Giroud, P., Ercsey-Ravasz, M., Pilaz, L.J., et al. (2011). Weight consistency
specifies the regularities of macaque cortical networks. Cereb. Cortex 21,
1254–1272.
- Markov, N.T., Ercsey-Ravasz, M.M., Ribeiro Gomes, A.R., Lamy, C., Magrou,
L., Vezoli, J., Misery, P., Falchier, A., Quilodran, R., Gariel, M.A., et al. (2012). A
weighted and directed interareal connectivity matrix for macaque cerebral
cortex. Cereb. Cortex. Published online September 25, 2012. http://dx.doi.
org/10.1093/cercor/bhs1270.
- Müllner, D. (2013). fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python. Journal of Statistical Software, 53(9), 1–18. https://doi.org/10.18637/jss.v053.i09

