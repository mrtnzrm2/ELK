import numpy as np

stirling_3 = lambda Nl, N : np.log((Nl/N) * (np.power(2*np.pi*Nl, 1/(2*Nl))/ np.power(2*np.pi*N, 1/(2*N)))) + (1/12)*(np.power(float(N), -2) - np.power(float(Nl), -2))
stirling_1 = lambda Nl, N : np.log(Nl/N)

class Hierarchical_Entropy:
  def __init__(self, Z, nodes : int, labels=[""]) -> None:
    from scipy.cluster.hierarchy import cut_tree
    self.total_nodes = nodes
    self.A = np.zeros((nodes, nodes))
    self.nodes = np.arange(nodes, dtype=int)
    self.height = np.zeros(nodes)
    self.A[nodes - 1, :] = self.nodes
    for i in np.arange(nodes - 1, 0, -1):
      self.A[i - 1, :] = cut_tree(Z, i).ravel()
    self.get_height_Z(Z)
    if len(labels) == nodes:
      self.labels = labels
    self.root = "L0_0"

  def get_height_Z(self, Z):
    self.height[1:] = Z[:, 2]

  def sum_vertices(self, tree : dict, i):
    for key in tree.keys():
      if key == "height" or key == "label": continue
      i += 1
      if len(tree[key]) == 1: continue
      self.sum_vertices(tree[key], i )

  def ML(self, tree : dict, ml : dict):
    for key in tree.keys():
      if key == "height" or key == "label": continue
      ski = key.split("_")[0]
      if ski not in ml.keys(): ml[ski] = 1
      else: ml[ski] += 1
      self.ML(tree[key], ml)

  def ML_height(self, tree : dict, ml : dict, maxlvl : int):
    for key in tree.keys():
      if key == "height" or key == "label": continue
      ski = key.split("_")[0]
      if self.is_leaf(tree[key]):
        if ski not in ml.keys():
          ml[ski] = {
            "size" : 1,
            "height" : 0 # tree[key]["height"]
          }
        else:
          ml[ski]["size"] += 1
          ml[ski]["height"] += 0 #tree[key]["height"]
      else:
        if ski not in ml.keys():
          ml[ski] = {
            "size" : 1,
            "height" : 0
          }
          for key2 in tree[key].keys():
            if key2 == "height" or key2 == "label": continue    
            ml[ski]["height"] += (tree[key]["height"] - tree[key][key2]["height"]) / 2
        else:
          ml[ski]["size"] += 1
          for key2 in tree[key].keys():
            if key2 == "height" or key2 == "label": continue    
            ml[ski]["height"] += (tree[key]["height"] - tree[key][key2]["height"]) / 2
      self.ML_height(tree[key], ml, maxlvl)

  def SH(self, tree : dict, Ml : dict, Sh):
    for key in tree.keys():
      if key == "height" or key == "label": continue
      if self.is_leaf(tree[key]): continue
      i = int(key.split("_")[0][1:])
      Mul = len(tree[key]) - 1
      Sh[self.total_nodes - i -1] -= Mul * stirling_3(Mul, Ml[f"L{i+1}"])
      self.SH(tree[key], Ml, Sh)

  def SV(self, Ml : dict, M, Sv):
    for key in Ml.keys():
      i = int(key[1:])
      Sv[self.total_nodes - i - 1] -= Ml[key] *  stirling_3(Ml[key], M)

  def is_leaf(self, tree : dict):
    if "label" in tree.keys():
      return True
    else: return False

  def SH_height(self, tree : dict, key, Ml : dict, maxl : int, Sh):
    i = int(key.split("_")[0][1:])
    if not self.is_leaf(tree[key]):
      Mul = np.sum([1 for key in tree[key].keys() if key != "height" and key != "label"])
      for key2 in tree[key].keys():
        if key2 == "height" or key2 == "label": continue
        Sh[self.total_nodes - i -1] -= (tree[key]["height"] - tree[key][key2]["height"]) * stirling_3(Mul, Ml[f"L{i+1}"]["size"]) / 2
        self.SH_height(tree[key], key2, Ml, maxl, Sh)

  def SV_height(self, Ml : dict, M, Sv):
    for key in Ml.keys():
      if key == "height" or key == "label": continue
      i = int(key[1:])
      Sv[self.total_nodes - i - 1] -= Ml[key]["height"] * stirling_3(Ml[key]["size"], M)

  def max_level(self, tree : dict, max_lvl):
    for key in tree.keys():
      if key == "height" or key == "label": continue
      ski = int(key.split("_")[0][1:])
      if max_lvl[0] < ski:
        max_lvl[0] = ski
      self.max_level(tree[key], max_lvl)

  def S(self, a):
    M = np.array([0])
    Ml = {}
    maxlevl = np.array([0])
    self.max_level(a, maxlevl)
    maxlevl = maxlevl[0]
    Sh =np.zeros(self.total_nodes)
    Sv = np.zeros(self.total_nodes)
    self.sum_vertices(a, M)
    self.ML(a, Ml)
    self.SH(a, Ml, Sh)
    self.SV(Ml, M, Sv)
    total_entropy = np.sum(Sh + Sv)
    Sh /= total_entropy
    Sv /= total_entropy
    print(f"\n\tNode entropy :  Sh : {np.sum(Sh):.4f}, and Sv : {np.sum(Sv):.4f}\n")
    return np.vstack([Sh[(self.total_nodes - maxlevl):], Sv[(self.total_nodes - maxlevl):]])
  
  def print_tree(self, a : dict, key_pred=""):
    for key in a.keys():
      if key == "height" or key == "label": continue
      # if len(a[key]) == 1:continue
      print(f"{key_pred}{key}", "\t\t", a[key]["height"])
      self.print_tree(a[key], key_pred=f"{key_pred}{key}")

  def print_tree_ml_h(self, a : dict):
    for key in a.keys():
      if key == "height" or key == "label": continue
      print(key, a[key]["size"], a[key]["height"])

  def print_tree_ml(self, a : dict):
    for key in a.keys():
      print(key, a[key])

  def S_height(self, a):
    # self.print_tree(a)
    M = np.array([0])
    Ml = {}
    maxlevl = np.array([0])
    self.sum_vertices(a, M)
    self.max_level(a, maxlevl)
    maxlevl = maxlevl[0]
    self.ML_height(a, Ml, maxlevl)
    # self.print_tree_ml_h(Ml)
    Sh = np.zeros(self.total_nodes)
    Sv = np.zeros(self.total_nodes)
    self.SH_height(a, self.root, Ml, maxlevl, Sh)
    self.SV_height(Ml, M, Sv)
    total_entropy_H = np.sum(Sv + Sh)
    Sh /= total_entropy_H
    Sv /= total_entropy_H
    print(f"\n\tNode entropy H: Sh : {np.sum(Sh):.4f}, and Sv : {np.sum(Sv):.4f}\n")
    return np.vstack([Sh[(self.total_nodes - maxlevl):], Sv[(self.total_nodes - maxlevl):]])

  def Z2dict_long(self, M, tree : dict, key_prev, nodes_prev : set, L, tL):
    if L < M.shape[0] and len(nodes_prev) > 1:
      coms = [M[L, i] for i in nodes_prev]
      for i, com in enumerate(np.unique(coms)):
        key = f"L{tL}_{int(com)}"
        nodes_com = set(list(np.where(M[L, :] == com)[0]))
        compare = nodes_com.intersection(nodes_prev)
        if len(compare) > 0:
          if key_prev not in tree.keys():
              tree[key_prev] = {
                "height" : self.height[self.total_nodes - L]
              }
          self.Z2dict_long(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
    else:
      if len(nodes_prev) == 1:
          tree[key_prev] = {
            "height" : self.height[self.total_nodes - L],
            "label" : self.labels[list(nodes_prev)[0]]
          }
      else:
        tree[key_prev] = {
          "height" : self.height[self.total_nodes - L]
        } 

  def Z2dict_short(self, M, tree : dict, key_prev, nodes_prev : set, L, tL):
    if L < M.shape[0] and len(nodes_prev) > 1:
      coms = [M[L, i] for i in nodes_prev]
      for i, com in enumerate(np.unique(coms)):
        key = f"L{tL}_{int(com)}"
        nodes_com = set(list(np.where(M[L, :] == com)[0]))
        compare = nodes_com.intersection(nodes_prev)
        if len(compare) == 0: continue
        if len(nodes_com) == len(nodes_prev):
          self.Z2dict_short(M, tree, key_prev, nodes_com, L + 1, tL)
        elif len(nodes_com) < len(nodes_prev):
          if key_prev not in tree.keys():
            tree[key_prev] = {
              "height" : self.height[self.total_nodes - L]
            }
          self.Z2dict_short(M, tree[key_prev], key, nodes_com, L + 1, tL + 1)
    else:
      if key_prev not in tree.keys():
        if len(nodes_prev) == 1:
          tree[key_prev] = {
            "height" : self.height[self.total_nodes - L],
            "label" : self.labels[list(nodes_prev)[0]]
          }
        else:
         tree[key_prev] = {
            "height" : self.height[self.total_nodes - L]
          } 

  def Z2dict(self, Z2):
    self.tree = {}
    nodes = set(list(self.nodes))
    L = 1
    tL = 1
    if Z2 == "short":
      self.Z2dict_short(self.A, self.tree, self.root, nodes, L, tL)
    elif Z2 == "long":
      self.Z2dict_long(self.A, self.tree, self.root, nodes, L, tL)
    else: raise ValueError("Only Z2 short or long")

  def zdict2pre_newick(self, tree : dict, root :str, key_pred : str, pre_newick : dict, weighted=False):
    if not self.is_leaf(tree[root]):
      leaves_names = [leaf for leaf in tree[root].keys() if leaf != "height" and leaf != "label"]
      is_root_leaf = [self.is_leaf(tree[root][leaf]) for leaf in leaves_names]
      distances = [tree[root]["height"] - tree[root][leaf]["height"] for leaf in leaves_names]
      if weighted:
        if np.sum(is_root_leaf) > 0: 
          dic1 = {key_pred + root + k: dis for is_leaf, k, dis in zip(is_root_leaf, leaves_names, distances) if not is_leaf}
          dic2 = {tree[root][leaf]["label"] : dis for is_leaf, leaf, dis in zip(is_root_leaf, leaves_names, distances) if is_leaf}
          pre_newick[key_pred + root] = {**dic1, **dic2}
        else:
          pre_newick[key_pred + root] = {key_pred + root + k: dis for k, dis in zip(leaves_names, distances)}
      else:
        if np.sum(is_root_leaf) > 0:
          dic1 = {key_pred + root + k: 1 for is_leaf, k in zip(is_root_leaf, leaves_names) if not is_leaf}
          dic2 = {tree[root][leaf]["label"] : 1 for is_leaf, leaf in zip(is_root_leaf, leaves_names) if is_leaf}
          pre_newick[key_pred + root] = {**dic1, **dic2}
        else:
          pre_newick[key_pred + root] = {key_pred + root + k: 1 for k in leaves_names}
      for leaves in leaves_names:
        self.zdict2pre_newick(tree[root], leaves, key_pred + root, pre_newick, weighted=weighted)

  def zdict2newick(self, tree, weighted=False, on=True):
    if on:
      print("Print hierarchy newicks")
      pre_newick = {}
      self.zdict2pre_newick(tree, self.root, "", pre_newick, weighted=weighted)
      newick = self.newickify(pre_newick, root_node=self.root)
      return newick

  def newickify(self, node_to_children, root_node) -> str:
    """Source code: https://stackoverflow.com/questions/50003007/how-to-convert-python-dictionary-to-newick-form-format"""
    visited_nodes = set()
    def newick_render_node(name, distance: float) -> str:
        assert name not in visited_nodes, "Error: The tree may not be circular!"
        if name not in node_to_children:
            # Leafs
            return F'{name}:{distance}'
        else:
            # Nodes
            visited_nodes.add(name)
            children = node_to_children[name]
            children_strings = [newick_render_node(child, children[child]) for child in children.keys()]
            children_strings = ",".join(children_strings)
            return F'({children_strings}):{distance}'
    newick_string = newick_render_node(root_node, 0) + ';'
    # Ensure no entries in the dictionary are left unused.
    assert visited_nodes == set(node_to_children.keys()), "Error: some nodes aren't in the tree"
    return newick_string
