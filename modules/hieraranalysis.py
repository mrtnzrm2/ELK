# Standard libs ----
import numpy as np
import pandas as pd
#  Personal libs ----
from various.network_tools import *
from various.discovery_channel import *
from modules.simanalysis import Sim
from modules.colregion import colregion
from process_hclust import ph
from la_arbre_a_merde import noeud_arbre
from h_entropy import h_entropy as HE

class Hierarchy(Sim):
  def __init__(
    self, G, A, R, nodes, linkage, mode, lookup=0, undirected=False
  ):
    # Initialize Sim ---
    super().__init__(
      nodes, A, R, mode,
      topology=G.topology, index=G.index,
      lookup=lookup, undirected=undirected
    )
    # Set parameters
    self.linkage = linkage
    self.cut = G.cut
    self.pickle_path = G.pickle_path
    self.plot_path = G.plot_path
    self.subfolder = G.subfolder
    self.analysis = G.analysis
    # Compute similarity matrix ----
    self.similarity_by_feature_cpp()
    # Compute distance matrix ----
    if self.index != "bsim":
      self.dist_mat = 1 - self.linksim_matrix
    else:
      self.dist_mat = self.linksim_matrix.copy()
      self.dist_mat[self.dist_mat != 0] -= np.nanmax(self.dist_mat[self.dist_mat != 0]) + 0.001
      self.dist_mat[self.dist_mat == 0] = np.nan
      self.dist_mat = np.nanmax(self.dist_mat) - self.dist_mat
      self.dist_mat[np.isnan(self.dist_mat)] = np.nanmax(self.dist_mat) + 1
    # Compute hierarchy ----
    self.H = self.get_hierarchy()
    self.delete_linksim_matrix()
    # Network to edgelist of EC ----
    non_x, non_y = np.where(self.nonzero[:self.nodes, :self.nodes])
    if not undirected:
       self.dA = pd.DataFrame(
        {
          "source" : list(non_x),
          "target" : list(non_y),
          "weight" : list(R[non_x, non_y])
        }
      )
    else:
      self.dA = pd.DataFrame(
        {
          "source" : list(non_x) + list(non_y),
          "target" : list(non_y) + list(non_x),
          "weight" : list(R[non_x, non_y]) * 2
        }
      )
    
    # Overlaps ----
    self.overlap = pd.DataFrame()
    # Cover ---
    self.cover = {}
    # KR ----
    self.kr = pd.DataFrame()
    # Entropy ----
    self.entropy = []
    # Discovery channel ----
    self.discovery_channel = discovery

  def delete_linksim_matrix(self):
    self.linksim_matrix = 0
    
  def delete_dist_matrix(self):
    self.dist_mat = 0

  def set_kr(self, k, r, score=""):
    self.kr = pd.concat(
      [
        self.kr,
        pd.DataFrame(
          {
            "K" : [k],
            "R" : [r],
            "score" : [score]
          }
        )
      ], ignore_index=True
    )

  def get_hierarchy(self):
    print("Compute link hierarchical agglomeration ----")
    from scipy.cluster.hierarchy import linkage
    return linkage(self.dist_mat, self.linkage)
  
  def H_features_cpp(self, linkage, cut=False):
    # Run process_hclust_fast.cpp ----
    features = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      linkage,
      cut,
      self.undirected
    )
    # features.bene("long")
    features.vite()
    result = np.array(
      [
        features.get_K(), features.get_Height(),
        features.get_NEC(),
        features.get_D(), features.get_ntrees(),
        features.get_X(), features.get_OrP(), features.get_XM(),
        features.get_S()
      ]
    )
    return result
  
  def H_features_cpp_nodewise(self, linkage, cut=False):
    # Run process_hclust_fast.cpp ----
    features = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      linkage,
      cut,
      self.undirected
    )
    # features.bene("long")
    k_equivalence = []
    k_equivalence.append(self.equivalence[0, 0])
    k_equivalence.append(1)
    for i in np.arange(self.equivalence.shape[0]-1):
      if self.equivalence[i, 1] != self.equivalence[i+1, 1]:
        k_equivalence.append(self.equivalence[i, 0])
    old = self.equivalence[0, 1]
    for i in np.arange(1, self.equivalence.shape[0]):
      if self.equivalence[i, 1] < old:
        k_equivalence.append(self.equivalence[i, 0])
        old = self.equivalence[i, 1]
    from collections import Counter
    count_r = dict(Counter(self.equivalence[:, 1]))
    count_r = {k : count_r[k] for k in -np.sort(-np.array(list(count_r.keys())))}
    t = 0
    for v in count_r.values():
      k_equivalence.append(self.equivalence[t + int(v/2), 0])
      t += int(v/2)
    k_equivalence = np.array(k_equivalence)
    k_equivalence = -np.sort(-np.unique(k_equivalence))
    features.vite_nodewise(
      k_equivalence, self.H[self.leaves - k_equivalence - 2, 2], k_equivalence.shape[0]
    )
    result = np.array(
      [
        features.get_K(), features.get_Height(),
        features.get_NEC(),
        features.get_D(), features.get_ntrees(),
        features.get_X(), features.get_OrP(), features.get_XM(),
        features.get_S()
      ]
    )
    return result
  
  
  def compute_H_features_cpp(self):
    print("\t> Compute features")
    # Set up linkage ----
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 2
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")

    results_no_mu = self.H_features_cpp(linkage, self.cut)

    self.FH = pd.DataFrame(
        {
          "K" : results_no_mu[0, :],
          "height" : results_no_mu[1, :],
          "NEC" : results_no_mu[2, :],
          "D" : results_no_mu[3, :],
          "ntrees": results_no_mu[4, :],
          "X" : results_no_mu[5, :],
          "m" : results_no_mu[6, :],
          "xm" : results_no_mu[7, :],
          "S" : results_no_mu[8, :],
          "SD" : (results_no_mu[3, :] / np.nansum(results_no_mu[3, :])) * (results_no_mu[8, :] / np.nansum(results_no_mu[8, :]))
        }
      )

  def compute_H_features_cpp_nodewise(self):
    print("\t> Compute features nodewise")
    # Set up linkage ----
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 2
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")
    
    results_nodewise = self.H_features_cpp_nodewise(linkage, self.cut)
    self.FH = pd.DataFrame(
      {
        "K" : results_nodewise[0, :],
        "height" : results_nodewise[1, :],
        "NEC" : results_nodewise[2, :],
        "D" : results_nodewise[3, :],
        "ntrees": results_nodewise[4, :],
        "X" : results_nodewise[5, :],
        "m" : results_nodewise[6, :],
        "xm" : results_nodewise[7, :],
        "S" : results_nodewise[8, :],
        "SD" : (results_nodewise[3, :] / np.nansum(results_nodewise[3, :])) * (results_nodewise[8, :] / np.nansum(results_nodewise[8, :]))
      }
    )

  def link_entropy_cpp(self, dist : str, cut=False):
    # from scipy.cluster.hierarchy import cut_tree
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 2
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")
    # Run process_hclust_fast.cpp ----
    entropy = ph(
      self.leaves,
      self.dist_mat,
      self.dA["source"].to_numpy()[:self.leaves],
      self.dA["target"].to_numpy()[:self.leaves],
      self.nodes,
      linkage,
      cut,
      self.undirected
    )

    entropy.arbre(dist)
    max_level = entropy.get_max_level()
    self.link_entropy = np.array(
      [entropy.get_entropy_h()[(self.leaves - max_level-1):], entropy.get_entropy_v()[(self.leaves - max_level-1):]]
    )
    total_entropy = np.sum(self.link_entropy)
    self.link_entropy = self.link_entropy / total_entropy
    self.link_entropy_H = np.array(
      [entropy.get_entropy_h_H()[(self.leaves - max_level-1):], entropy.get_entropy_v_H()[(self.leaves - max_level-1):]]
    )
    total_entropy_H = np.sum(self.link_entropy_H)
    self.link_entropy_H = self.link_entropy_H / total_entropy_H
    sh = np.nansum(self.link_entropy[0, :])
    sv = np.nansum(self.link_entropy[1, :])
    print(f"\n\tlink entropy :  Sh : {sh:.4f}, and Sv : {sv:.4f}\n")
    sh = np.nansum(self.link_entropy_H[0, :])
    sv = np.nansum(self.link_entropy_H[1, :])
    print(f"\n\tlink entropy H: Sh : {sh:.4f}, and Sv : {sv:.4f}\n")

  
  def node_entropy_cpp(self, dist : str, cut=False):
    # from scipy.cluster.hierarchy import cut_tree
    # Run process_hclust_fast.cpp ----
    entropy = HE(self.Z, self.nodes)
    entropy.arbre(dist)
    max_level = entropy.get_max_level()
    self.node_entropy = np.array(
      [entropy.get_entropy_h()[(self.nodes - max_level-1):], entropy.get_entropy_v()[(self.nodes - max_level-1):]]
    )
    total_entropy = np.sum(self.node_entropy)
    self.node_entropy = self.node_entropy / total_entropy
    self.node_entropy_H = np.array(
      [entropy.get_entropy_h_H()[(self.nodes - max_level-1):], entropy.get_entropy_v_H()[(self.nodes - max_level-1):]]
    )
    total_entropy_H = np.sum(self.node_entropy_H)
    self.node_entropy_H = self.node_entropy_H / total_entropy_H
    sh = np.nansum(self.node_entropy[0, :])
    sv = np.nansum(self.node_entropy[1, :])
    print(f"\n\tNode entropy :  Sh : {sh:.4f}, and Sv : {sv:.4f}\n")
    sh = np.nansum(self.node_entropy_H[0, :])
    sv = np.nansum(self.node_entropy_H[1, :])
    print(f"\n\tNode entropy H: Sh : {sh:.4f}, and Sv : {sv:.4f}\n")


  def la_abre_a_merde_cpp(self, sp=25):
    print("\t> Compute the node hierarchy ----")
    # Get network dataframe ----
    dA =  self.dA.copy()
     # Set up linkage ----
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 1
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")
    # print(self.FH)
    # Run la_abre_a_merde_vite ----
    # if self.FH.K.iloc[-1] != 1:
    #   self.FH = pd.concat(
    #     [
    #       self.FH,
    #       pd.DataFrame(
    #         {
    #           "K" : 1,
    #           "height" : [self.FH.height.iloc[-1] * 1.01],
    #           "NEC" : [1]
    #         }
    #       )
    #     ],
    #     ignore_index=True
    #   )
    NH = noeud_arbre(
      self.dist_mat,
      dA["source"].to_numpy(),
      dA["target"].to_numpy(),
      self.FH["K"].to_numpy().astype(int),
      self.FH["height"].to_numpy(),
      self.FH["NEC"].to_numpy().astype(int),
      self.nodes,
      self.leaves,
      linkage,
      self.FH.shape[0],
      sp,
      self.undirected
    )
    self.Z = NH.get_node_hierarchy()
    self.Z = np.array(self.Z)
    self.equivalence = NH.get_equivalence()
    self.equivalence = np.array(self.equivalence)

  def la_abre_a_merde_cpp_no_feat(self, sp=25):
    print("\t> Compute node hierarchy no feat ----")
    # Get network dataframe ----
    dA =  self.dA.copy()
     # Set up linkage ----
    if self.linkage == "single":
      linkage = 0
    elif self.linkage == "average":
      linkage = 1
    else:
      linkage = -1
      raise ValueError("Link community model has not been tested with the input linkage.")
    # Run la_abre_a_merde_vite ----
    NH = noeud_arbre(
      self.dist_mat,
      dA["source"].to_numpy(),
      dA["target"].to_numpy(),
      np.arange(self.leaves - 1, 0, -1, dtype=int),
      self.H[:, 2].ravel(),
      np.array([1] * (self.leaves - 1)),
      self.nodes,
      self.leaves,
      linkage,
      self.leaves - 1,
      sp,
      self.undirected
    )
    self.Z = NH.get_node_hierarchy()
    self.Z = np.array(self.Z)
    self.equivalence = NH.get_equivalence()
    self.equivalence = np.array(self.equivalence)

  def set_colregion(self, colregion : colregion):
    self.colregion = colregion

  def set_overlap_labels(self, labels, score):
    subdata = pd.DataFrame(
      {
        "labels": labels,
        "score" : [score] * len(labels)
      }
    )
    self.overlap = pd.concat(
      [self.overlap, subdata],
      ignore_index=True
    )
  
  def set_cover(self, cover, score):
    self.cover[score] = cover