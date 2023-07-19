import numpy as np
import pandas as pd
from various.network_tools import *

def cover_assignment(H, K : int, Cr, labels, undirected=False, **kwargs):  
  from itertools import combinations
  from scipy.cluster.hierarchy import cut_tree
  nocs_size = {}
  nocs = {}
  overlap = np.zeros(H.nodes)
  skimCr = skim_partition(Cr)
  dA = H.dA.copy()
  ## Cut tree ----
  if not undirected:
    dA["id"] = cut_tree(H.H, n_clusters=K).ravel()
  else:
    dA["id"] = np.tile(cut_tree(H.H, n_clusters=K).ravel(), 2)
  minus_one_Dc(dA, undirected)
  aesthetic_ids(dA)
  Dsource = 1 / H.source_sim_matrix + 1
  Dtarget = 1 / H.target_sim_matrix + 1
  Dsource[Dsource == np.Inf] = np.max(Dsource[Dsource < np.Inf])
  Dtarget[Dtarget == np.Inf] = np.max(Dtarget[Dtarget < np.Inf])
  m_s = np.nanmax(Dsource)
  m_t = np.nanmax(Dtarget)
  max_m = np.sqrt(np.power(m_s, 2) + np.power(m_t, 2))
  ## Single nodes ----
  single_nodes = [np.where(Cr == i)[0][0] for i in np.unique(Cr) if np.sum(Cr == i) == 1]
  ## Nodes with single community membership ----
  NSC = [(set(np.where(skimCr == i)[0]), i) for i in np.unique(skimCr) if i != -1]
  if len(NSC) > 0:
    for sn in single_nodes:
      dsn_src = dA.loc[dA.source == sn]
      dsn_tgt = dA.loc[dA.target == sn]
      Dsn = np.zeros((len(NSC), 2))
      for ii, nsc in enumerate(NSC):
        neighbor_nodes_src = set(dsn_src.target).intersection(nsc[0])
        neighbor_nodes_tgt = set(dsn_tgt.source).intersection(nsc[0])
        neighbors = list(neighbor_nodes_src.union(neighbor_nodes_tgt))
        if len(neighbors) > 0:
          Dsn[ii, 0] = np.nanmin(Dsource[sn, neighbors])
          Dsn[ii, 1] = np.nanmin(Dtarget[sn, neighbors])
      if len(NSC) > 1:
        comb = [(u, v) for u, v in combinations(range(len(NSC)), 2)]
      else: comb = [(0, 0)]
      Dsn[Dsn == 0] = np.nan
      Rc = np.nanmean(Dsn, axis=0)
      Dsn[np.isnan(Dsn[:, 0]), 0] = m_s
      Dsn[np.isnan(Dsn[:, 1]), 1] = m_t
      Rs = np.array([np.linalg.norm(Dsn[u] - Dsn[v]) for u, v in comb if np.linalg.norm(Dsn[u]) < max_m and np.linalg.norm(Dsn[v]) < max_m])
      if len(Rs) > 0:
        Rs = np.argmin(Rs)
      else: Rs = 0
      Rs = np.abs(Dsn[comb[Rs][0]] - Dsn[comb[Rs][1]])
      Rc = Rc + Rs / (len([i for i in range(len(NSC)) if np.linalg.norm(Dsn[i]) < max_m]))

      Rc = np.linalg.norm(Rc)
      Dsn = np.linalg.norm(Dsn, axis=1)
      for ii, nsc in enumerate(NSC):
        if not np.isnan(Dsn[ii]):
          if Dsn[ii] <= Rc and Dsn[ii] < max_m:
            if labels[sn] not in nocs.keys():
              nocs[labels[sn]] = [nsc[1]]
              nocs_size[labels[sn]] = {nsc[1] : np.exp(-Dsn[ii]/max_m)}
            else:
              nocs[labels[sn]].append(nsc[1])
              nocs_size[labels[sn]].update({nsc[1] : np.exp(-Dsn[ii]/max_m)})
            overlap[sn] += 1
  # for k in nocs.keys(): nocs[k] = list(np.unique(nocs[k]))
  for sn in single_nodes:
    if labels[sn] not in nocs.keys():
      overlap[sn] += 1
  return np.where(overlap > 0)[0], nocs, nocs_size
  
def discovery(H, K : int, Cr, undirected=False, **kwargs):
  labels = H.colregion.labels[:H.nodes]
  overlap, noc_covers, noc_sizes = cover_assignment(H, K, Cr, labels, undirected=undirected, **kwargs)
  return np.array([labels[i] for i in overlap]), noc_covers, noc_sizes