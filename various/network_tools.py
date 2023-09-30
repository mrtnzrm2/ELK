import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from os.path import join, exists, isfile
from os import remove, stat
import pickle as pk
from various.omega import Omega

def skim_partition(partition):
  par = partition.copy()
  from collections import Counter
  fq = Counter(par)
  for i in fq.keys():
    if fq[i] == 1: par[par == i] = -1
  new_partition = par
  ndc = np.unique(par[par != -1])
  for i, c in enumerate(ndc):
    new_partition[par == c] = i
  return new_partition

def Dc_id(dA, id, undirected=False):
  # Filter dataframe ----
  if not undirected:
    dAid = dA.loc[dA.id == id]
  else:
    dAid = dA.loc[(dA.id == id) & (dA.source > dA.target)]
  # Get source nodes ----
  src = set([i for i  in dAid.source])
  # Get target nodes list ----
  tgt = set([i for i in dAid.target])
  # Get number of edges ----
  m = dAid.shape[0]
  # Compute Dc ----
  n = len(tgt.union(src))
  if ~undirected:
    if n > 1 and m >= n: return (m - n + 1) / (n - 1) ** 2
    else: return 0
  else:
   if n > 1 and m >= n: return (m - n + 1) / (n * (n - 1) / 2. - n + 1.)
   else: return 0

def minus_one_Dc(dA, undirected=False):
  ids = np.sort(
    np.unique(
      dA["id"].to_numpy()
    )
  )
  for id in ids:
    Dc = Dc_id(dA, id, undirected=undirected)
    if Dc <= 0:
      dA["id"].loc[dA["id"] == id] = -1

def get_best_kr(score, H, undirected=False, mapping="X_diag"):
  k = 1
  if score == "_D":
    k = get_k_from_D(H.FH)
    k = [k]
  elif score == "_X":
    k = get_k_from_X(H.FH, order=0)
    k = [k]
  elif score == "_S":
    k = get_k_from_S(H.FH)
  elif score == "_SD":
    k = get_k_from_SD(H.FH)
  else: raise ValueError(f"Unexpected score: {score}")
  if not isinstance(k, list): k = [k]
  r = get_r_from[mapping](
    k, H.H, H.Z, H.A, H.nodes, undirected=undirected
  )
  if isinstance(r, list) and isinstance(k, list):
    return np.array(k), np.array(r)
  elif isinstance(k, list):
    return np.array(k), np.array([r])
  elif isinstance(r, list):
    return np.array([k]), np.array(r)
  else:
    return np.array([k]), np.array([r])

def get_best_kr_equivalence(score, H):
  k = 1
  if score == "_D":
    k = get_k_from_D(H.FH)
  elif score == "_X":
    k = get_k_from_X(H.FH, order=0)
  elif score == "_SL":
    k = get_k_from_S(H.FH)
  elif score == "_SD":
    k = get_k_from_SD(H.FH)
  else: raise ValueError(f"Unexpected score: {score}")
  r = get_r_from_equivalence(k, H)
  return k, r

def aesthetic_ids(dA):
    ids = np.sort(
      np.unique(
        dA["id"].to_numpy()
      )
    )
    if -1 in ids:
      ids = ids[1:]
      aids = np.arange(1, len(ids) + 1)
    else:
      aids = np.arange(1, len(ids) + 1)
    for i, id in enumerate(ids):
      dA.loc[dA["id"] == id, "id"] = aids[i].astype(str)
    dA["id"] = dA["id"].astype(int)

def combine_dics(f1, f2):
  for k in f2.keys():
    if k not in f1.keys():
      f1[k] = f2[k]
    else:
      f1[k] += f2[k]

def reverse_partition(Cr, labels):
  s = np.unique(Cr).astype(int)
  s = s[s != -1]
  k = {r : [] for r in s}
  for i, r in enumerate(Cr):
    if r == -1: continue
    k[r].append(labels[i])
  return k

def nocs2parition(partition: dict, nocs: dict):
  for noc in nocs.keys():
    for cover in nocs[noc]:
      if str(noc) not in partition[cover]:
        partition[cover].append(str(noc))

def get_k_from_X(H, order=0):
  r = H["K"].loc[
    H["X"] == np.nanmax(H["X"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return int(r)

def get_labels_from_Z(Z, r):
  save_Z = np.sum(Z, axis=1)
  if 0 in save_Z: return np.array([np.nan])
  from scipy.cluster.hierarchy import cut_tree
  labels = cut_tree(
    Z,
    n_clusters=r
  ).reshape(-1)
  return labels

def get_k_from_D(H):
  r = H["K"].loc[
    H["D"] == np.nanmax(H["D"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return int(r)

def get_k_from_S(H):
  r = H["K"].loc[
    H["S"] == np.nanmax(H["S"])
  ]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return int(r)

def get_k_from_SD(H):
  r = H["K"].loc[H.SD == np.nanmax(H.SD)]
  if (len(r) > 1):
    print("warning: more than one k")
    r = r.iloc[0]
  return int(r)

def get_r_from_equivalence(k, H):
  return H.equivalence[H.equivalence[:, 0] == k, 1][0]

def get_r_from_X_diag(K, H, Z, R, nodes, **kwargs):
  from scipy.cluster.hierarchy import cut_tree, dendrogram
  r = []
  for k in K:
    labels = cut_tree(H, k).ravel()
    dR = adj2df(R[:nodes, :])
    dR = dR.loc[(dR.weight != 0)]
    dR["id"] = labels
    minus_one_Dc(dR, False)
    aesthetic_ids(dR)
    ##
    den_order = np.array(
      dendrogram(Z, no_plot=True)["ivl"]
    ).astype(int)
    RR = df2adj(dR, var="id")[:, den_order][den_order, :]
    dR = adj2df(RR)
    dR["id"] = dR.weight
    dR = dR.loc[(dR.id != 0)]
    ##
    unique_labels = np.unique(dR.id)
    len_unique_labels = 0
    dR = dR.loc[dR.id != -1]
    for label in unique_labels:
      if label == -1: continue
      dr_down = dR.source.loc[(dR.id == label) & (dR.target < dR.source)]
      dr_up = dR.target.loc[(dR.id == label) & (dR.source < dR.target)]
      len_between_up_down = len(set(dr_up).intersection(set(dr_down)))
      if len_between_up_down > 0:
        len_unique_labels += 1
      else:
        dR = dR.loc[dR.id != label]
    nodes_fair_communities = set(dR.source).intersection(set(dR.target))
    r.append(nodes - len(nodes_fair_communities) + len_between_up_down)
  return r

def get_r_from_modularity(k, H, Z, R, nodes, undirected=False):
  from collections import Counter
  from scipy.cluster.hierarchy import cut_tree, dendrogram
  if not undirected:
    RR = R.copy()
    RR[RR != 0] = 1
    RR = RR[:nodes, :]
    RR = adj2df(RR)
    RR = RR.loc[RR.weight != 0]
    RR["id"] = cut_tree(H, k).ravel()
  else:
    RR = np.triu(R)
    nonzero = RR != 0
    nonx, nony = np.where(nonzero)
    RR = pd.DataFrame(
      {
        "source" : list(nonx) + list(nony),
        "target" : list(nony) + list(nonx),
        "weight" : list(R[nonx, nony]) * 2
      }
    )
    RR["id"] = np.tile(cut_tree(H, k).ravel(), 2)
  minus_one_Dc(RR, undirected=undirected)
  aesthetic_ids(RR)
  RR2 = df2adj(RR, var="id")
  RR2[RR2 == -1] = 0
  RR2[RR2 != 0] = 1
  RR = df2adj(RR)
  RR = RR * RR2
  #
  den_order = np.array(dendrogram(Z, no_plot=True)["ivl"]).astype(int)
  RR = RR[den_order, :][:, den_order]
  D = np.zeros(nodes - 1)
  for i in np.arange(nodes - 1, 0, -1):
    partition = cut_tree(Z, i).ravel()[den_order]
    number_nodes = dict(Counter(partition))
    where_nodes = {k : np.where(partition == k)[0] for k in number_nodes.keys()}
    d = np.array([(number_nodes[k] / nodes) * np.nansum(RR[where_nodes[k], :][:, where_nodes[k]]) / (number_nodes[k] * (number_nodes[k] - 1)) for k in number_nodes.keys()])
    D[i-1] = np.nansum(d)
  return np.argmax(D)

get_r_from = {
  "modularity" : get_r_from_modularity,
  "X_diag" : get_r_from_X_diag
}

def AD_NMI_label(gt, pred, on=True):
  if on:
    if np.sum(np.isnan(pred)) > 0: nmi = np.nan
    elif len(np.unique(pred)) == 1: nmi = np.nan
    else:
      from sklearn.metrics import adjusted_mutual_info_score
      nmi = adjusted_mutual_info_score(gt, pred, average_method="max")
    print("ADNMI: {}".format(nmi))
    return nmi

def save_class(
  CLASS, pickle_path, class_name="duck", on=True, **kwargs
):
  path = join(
    pickle_path, "{}.pk".format(class_name)
  )
  if exists(path): remove(path)
  if on:
    with open(path, "wb") as f:
      pk.dump(CLASS, f)

def read_class(pickle_path, class_name="duck", **kwargs):
  path = join(
    pickle_path, "{}.pk".format(class_name)
  )
  C = 0
  print(path)
  if isfile(path) and stat(path).st_size > 100:
    with open(path, "rb") as f:
      C =  pk.load(f)
  else: print(f"\nFile {path} does not exist\n")
  return C

def column_normalize(A):
  if np.sum(np.isnan(A)) > 0:
    raise ValueError("\nColumn normalied does not accept nan. Use instead column_normalize_nan.\n")
  C = A.sum(axis = 0)
  C = A.copy() / C
  C[np.isnan(C)] = 0
  return C

def column_normalize_nan(A):
  C = np.nansum(A, axis=0)
  C = A.copy() / C
  C[np.isnan(C)] = 0
  return C

def match(a, b):
    b_dict = {x: i for i, x in enumerate(b)}
    return np.array([b_dict.get(x, None) for x in a])
    
def sort_by_size(ids, nodes):
  # Define location memory ---
  c = 0
  # Define new id ----
  nids = np.zeros(nodes)
  # Find membership frequency ----
  from collections import Counter
  f = Counter(ids)
  f = dict(f)
  f = sort_dict_value(f)
  for key in f:
    w = np.where(ids == key)[0]
    lw = len(w)
    nids[c:(lw + c)] = w
    c += lw
  return nids.astype(int), f

def sort_dict_value(counter : dict):
  f = {
    k: v for k, v in sorted(
      counter.items(), key=lambda item: item[1],
      reverse=True
    )
  }
  return f

def invert_dict_single(f: dict):
  return {v : k for k, v in f.items()}

def invert_dict_multiple(f : dict):
  ff = {}
  for key in f.keys():
    for val in f[key]:
      if val not in ff.keys(): ff[val] = [key]
      else: ff[val] = ff[val] + [key]
  for key in ff.keys(): ff[key] = list(np.unique(ff[key]))
  return ff

def membership2ids(Cr, dA):
  skimCr = skim_partition(Cr)
  # uCr = np.unique(Cr)
  uCr = np.unique(skimCr[skimCr != -1])
  nodecom_2_id = {
    ur : [] for ur in uCr
  }
  for ur in uCr:
    ur_nodes = np.where(Cr == ur)[0]
    if len(ur_nodes) == 0: continue
    dur = np.unique(dA.id.loc[np.isin(dA.source, ur_nodes)])
    nodecom_2_id[ur] = nodecom_2_id[ur] + list(dur)
    dur = np.unique(dA.id.loc[np.isin(dA.target, ur_nodes)])
    nodecom_2_id[ur] = nodecom_2_id[ur] + list(dur)
  for key in nodecom_2_id:
    nodecom_2_id[key] = list(np.unique(nodecom_2_id[key]))
  return nodecom_2_id

def condense_madtrix(A):
  n = A.shape[0] * (A.shape[0] - 1) / 2
  cma = np.zeros(int(n))
  t = 0
  for i in np.arange(A.shape[0] - 1):
    for j in np.arange(i + 1, A.shape[0]):
      cma[t] =  A[i, j]
      t += 1
  return cma

def df2adj(dA, var="weight"):
  m = np.max(dA["source"].to_numpy()) + 1
  n = np.max(dA["target"].to_numpy()) + 1
  A = np.zeros((m.astype(int), n.astype(int)))
  A[
    dA["source"].to_numpy().astype(int),
    dA["target"].to_numpy().astype(int)
  ] = dA[var].to_numpy()
  return A

def adj2df(A):
  src = np.repeat(
    np.arange(A.shape[0]),
    A.shape[1]
  )
  tgt = np.tile(
    np.arange(A.shape[1]),
    A.shape[0]
  )
  dA = pd.DataFrame(
    {
      "source" : src,
      "target" : tgt,
      "weight" : A.reshape(-1)
    }
  )
  return dA

def omega_index_format(node_partition, noc_covers : dict, node_labels):
  rev = reverse_partition(node_partition, node_labels)
  nocs2parition(rev, noc_covers)
  return rev

def reverse_cover(cover: dict, labels):
  cover_indices = set()
  for k, v in cover.items():
    cover_indices = cover_indices.union(set(v))
  rev = {k : [] for k in cover_indices}
  for k, v in cover.items():
    for vv in v:
      rev[vv].append(labels[k])
  return rev

def omega_index(cover_1 : dict, cover_2 : dict):
  if len(cover_1) == 1 and len(cover_2) == 1:
    omega = np.nan
  else:
    omega = Omega(cover_1, cover_2).omega_score
  print(f"Omega: {omega:.4f}")
  return omega
    
