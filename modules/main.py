# Python libs ----
import numpy as np
import pandas as pd
#  ELK libraries ----
from various.network_tools import *
from modules.simanalysis import Sim
from modules.colregion import colregion
from process_hclust import ph
from la_arbre_a_merde import noeud_arbre
from h_entropy import h_entropy as HE

class ELK(Sim):
  def __init__(
    self, R, nodes, linkage, mode, topology="MIX", index="S1_2", cut=False, lookup=0, undirected=False, **kwargs
  ):
    # Initialize Sim ---
    super().__init__(
      nodes, R, mode,
      topology=topology, index=index,
      lookup=lookup, undirected=undirected
    )
    # Set parameters
    self.linkage = linkage
    self.cut = cut
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
    self.entropy = []

  def delete_linksim_matrix(self):
    self.linksim_matrix = 0
    
  def delete_dist_matrix(self):
    self.dist_mat = 0

  def set_entropy(self, entropies):
    self.entropy = entropies

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

    ento = entropy.get_entropy_h()[(self.leaves - max_level-1):]
    ento_h = entropy.get_entropy_v()[(self.leaves - max_level-1):]
    self.link_entropy = np.array(
      [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
    )
    total_entropy = np.sum(self.link_entropy)
    self.link_entropy = self.link_entropy / total_entropy

    ento = entropy.get_entropy_h_H()[(self.leaves - max_level-1):]
    ento_h = entropy.get_entropy_v_H()[(self.leaves - max_level-1):]
    self.link_entropy_H = np.array(
       [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
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
    entropy = HE(self.Z, self.nodes)
    entropy.arbre(dist)
    max_level = entropy.get_max_level()

    ento = entropy.get_entropy_h()[(self.nodes - max_level-1):]
    ento_h = entropy.get_entropy_v()[(self.nodes - max_level-1):]
    self.node_entropy = np.array(
      [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
    )
    total_entropy = np.sum(self.node_entropy)
    self.node_entropy = self.node_entropy / total_entropy

    ento = entropy.get_entropy_h_H()[(self.nodes - max_level-1):]
    ento_h = entropy.get_entropy_v_H()[(self.nodes - max_level-1):]
    self.node_entropy_H = np.array(
      [np.hstack([ento[1:], ento[0]]), np.hstack([ento_h[1:], ento_h[0]])]
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

  def get_nocs_information(self, Cr, dA, labels, direction, index):
  
    from scipy.cluster.hierarchy import linkage, cut_tree
    from scipy.spatial.distance import squareform

    nocs_size = {}
    nocs = {}
    overlap = np.zeros(self.nodes)

    if index == "S1_2":
      if direction == "source":
        np.seterr(divide='ignore', invalid='ignore')
        Diss = (1 / self.source_sim_matrix) - 1
        Diss[Diss == np.Inf] = np.max(Diss[Diss < np.Inf]) * 1.05
      elif direction == "target":
        np.seterr(divide='ignore', invalid='ignore')
        Diss = (1 / self.target_sim_matrix) - 1
        Diss[Diss == np.Inf] = np.max(Diss[Diss < np.Inf]) * 1.05
      else:
        raise ValueError("No accepted direction in discovery channel")
    elif index == "H2":
      if direction == "source":
        Diss = np.sqrt(1 - self.source_sim_matrix)
      elif direction == "target":
        Diss = np.sqrt(1 - self.target_sim_matrix)
      else:
        raise ValueError("No accepted direction in discovery channel")
    else: raise ValueError("No accepeted index in discovery channel")

    th = np.max(Diss) * 1.05
    
    ## Single nodes ----
    single_nodes = np.where(Cr == -1)[0]
    ## Nodes with single community membership ----
    NSC = [(set(np.where(Cr == i)[0]), i) for i in np.unique(Cr) if i != -1]

    for sn in single_nodes:
      if direction == "source":
        dsn = set(dA.loc[dA.source == sn].target)
      elif direction == "target":
        dsn = set(dA.loc[dA.target == sn].source)
      else:
        dsn1 = set(dA.loc[dA.target == sn].source)
        dsn2 = set(dA.loc[dA.source == sn].target)
        dsn = dsn1.intersection(dsn2)

      Dsn = np.zeros((len(NSC)))

      for ii, nsc in enumerate(NSC):
        neighbor_nodes = list(dsn.intersection(nsc[0]))

        if len(neighbor_nodes) > 0:
          Dsn[ii] = np.mean(Diss[sn, neighbor_nodes])
        else:
          Dsn[ii] = th

      non_trivial_covers = Dsn < th

      if np.sum(non_trivial_covers) > 0:

        nn = Dsn[non_trivial_covers].shape[0]
        indx_min = np.argmin(Dsn[non_trivial_covers])
        dsn_min = Dsn[non_trivial_covers][indx_min]

        if nn > 1:
          DD = np.zeros((nn, nn))
          for kk in np.arange(nn):
            for ki in np.arange(kk+1, nn):
              DD[kk, ki] = np.abs(Dsn[non_trivial_covers][kk] - Dsn[non_trivial_covers][ki])
              DD[ki, kk] = DD[kk, ki]

          DD = linkage(squareform(DD), method="complete")
          h = np.argmax(DD[:, 2])
          li = cut_tree(DD, height=DD[h-1, 2]).ravel()
          min_point_region = li[indx_min]

        else:
          li = [0]
          min_point_region =  0

        ii = 0
        for nsc, non in zip(NSC, non_trivial_covers):
          if non:
            if li[ii] == min_point_region or Dsn[non_trivial_covers][ii] == dsn_min:
              if labels[sn] not in nocs.keys():
                nocs[labels[sn]] = [nsc[1]]
                nocs_size[labels[sn]] = {nsc[1] : th - Dsn[non_trivial_covers][ii]}

              else:
                nocs[labels[sn]].append(nsc[1])
                nocs_size[labels[sn]].update({nsc[1] : th - Dsn[non_trivial_covers][ii]})
              overlap[sn] += 1
            ii += 1

    return  np.array(list(nocs.keys())), nocs, nocs_size

  def cover_assignment(self, Cr, K, labels, undirected=False, direction="source", index="H2", **kwargs):  
    from scipy.cluster.hierarchy import cut_tree
    cr = Cr.copy()

    dA = self.dA.copy()
    ## Cut tree ----
    if not undirected:
      dA["id"] = cut_tree(self.H, n_clusters=K).ravel()
    else:
      dA["id"] = np.tile(cut_tree(self.H, n_clusters=K).ravel(), 2)

    if direction == "source":
      overlap, nocs, nocs_size = self.get_nocs_information(Cr, dA, labels, "source", index)
    elif direction == "target":
      overlap, nocs, nocs_size = self.get_nocs_information(Cr,dA, labels, "target", index)
    elif direction == "both":
      overlap_src, nocs_src, nocs_size_src = self.get_nocs_information(Cr, dA, labels, "source", index)
      overlap_tgt, nocs_tgt, nocs_size_tgt = self.get_nocs_information(Cr, dA, labels, "target", index)

      overlap = np.hstack([overlap_src, overlap_tgt])
      overlap = np.unique(overlap)

      nocs = nocs_src.copy()

      for key, value in nocs_tgt.items():
        if key not in nocs.keys():
          nocs[key] = value
        else:
          nocs[key] += value
          nocs[key] = list(set(nocs[key]))

      nocs_size = nocs_size_src.copy()

      for key, value in nocs_size_tgt.items():
        if key not in nocs_size.keys():
          nocs_size[key] = value
        else:
          for key2, value2 in nocs_size_tgt[key].items():
            if key2 not in nocs_size[key].keys():
              nocs_size[key].update({key2 : value2})
            else:
              nocs_size[key][key2] = 0.5 * (value2 + nocs_size[key][key2])

    not_nocs = []

    for key in nocs.keys():
      if len(nocs[key]) == 1:
        not_nocs.append(key)
      i = match([key], labels)
      if len(nocs[key]) == 1 and cr[i] == -1:
        cr[i] = nocs[key][0]

    for key in not_nocs:
      del nocs[key]
      del nocs_size[key]

    return np.array(list(nocs.keys())), nocs, nocs_size, cr
    
  def discovery(self, Cr, K, undirected=False, direction="both", index='H2', **kwargs):
    labels = self.colregion.labels[:self.nodes]
    overlap, noc_covers, noc_sizes, new_partition = self.cover_assignment(
      Cr, K, labels, undirected=undirected, direction=direction, index=index, **kwargs
    )
    return overlap, noc_covers, noc_sizes, new_partition