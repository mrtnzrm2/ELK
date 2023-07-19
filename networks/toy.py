# Standard libs ----
from pathlib import Path
from os.path import join
import numpy as np
# Personalized libs ----
from networks.base import BASE

class TOY(BASE):
  """
    Class to process the network in the format needed for the Hierarchy class to work.
    Hierarchy was created to use specifically in the macaque FLN network.
    **In the future, there will be another class to run the link community algorithm
    in particular netowrks.** For the moment, the network has to be directed.
    
    Parameters:
        A: An NxM directed adjacency matrix.
          The algorithm can compute similarities using
          the whole data, but only classsify links from the edge-comple graph.

        linkage: string.
          Desired linkage-method for the hierarchical agglomeration algorithm.
          Currenly, only working with the single and average linkage method.

        nlog10: bool.
          If true, the data has to be transformed using a logarithmic transformation.
          It also adds a "_l10" suffix to the plot and pickle directory.

        lookup: bool.
          If true, the zeros in the network will be assigned to a value coming from
          the applied transformation of the data. Ex, in case of choosing the R1, or
          inverse logarithmic trasnformation, the zeros in the network will be assigned
          the maximum value of the new range. This modification will only contribute for
          computing the similarities. They do not modify the topology of the network. It
          adds a "_lup" suffix to the plot and pickle directory.

        cut: bool.
          If true, it tells the link community algorithm if it can skip link community
          merging steps where the distance between link communities is constant.
          It skips those steps unitl the algorithm merges two communities with
          a new distance. It helps to reduce the computing time of the algorithm without
          getting results very different from full analysis. It adds as "_cut" suffix
          to the plot and pickle directory.

        mapping: string.
          Type of mapping applied to the network. There are three types.
              >> "trivial": No transformation
              >> "R1" : Inverse logarithmic transformation, i.e., applying x <- -log10(x)
              >> "R2" : Normal logarithmic transformation, i.e., applying x <-  log10(x) + b
        
        index: string.
          Type of similarity index. Currently, there are three types:
            >> "jacp" : Jaccard probability index
            >> "cos" : Cosine similarity
            >> "jacw" : Modified weighted Jaccard index

        topology: string.
          Type of node neighborhoods to consider which could be:
            >> "MIX" : Source similarities for in-links and target
                similarities for out-links.
            >> "SOURCE : Source similarities to in- and out-links.
            >> "TARGET" : Target similarities to in- and out-links.
            
  """
  def __init__(self, A, **kwargs) -> None:
    super().__init__(**kwargs)
    self.A = A.copy()
    self.nodes, self.rows = A.shape
    # Overlap ----
    self.overlap = np.array(["UNKNOWN"] * self.nodes)
    # NOCS dict ----
    self.data_nocs = {}

  def set_labels(self, labels):
    self.labels = labels
    self.struct_labels = labels