# Python libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
import os
# ELK libs ----
from modules.hieraranalysis import Hierarchy
from various.network_tools import *

class Plot_H:
  def __init__(self, H : Hierarchy) -> None:
    ## Attributes ----
    self.leaves= H.leaves
    self.linkage = H.linkage
    self.FH = H.FH
    self.Z = H.Z
    self.H = H.H
    self.A = H.A
    self.nonzero = H.nonzero
    self.dA = H.dA
    self.nodes = H.nodes
    self.mode = H.mode
    self.leaves = H.leaves
    self.index = H.index
    self.entropy = H.entropy
    # Get regions and colors ----
    self.colregion = H.colregion
    self.labels = H.colregion.labels[:self.nodes]

  def plot_Hierarchical_Entropy(self, width=8, height=6):
    print("\t> Visualize Hierarchical Entropy by levels!!!")
    # Create data ----
    dim = self.entropy[0].shape[1]
    print(f"Levels node hierarchy: {dim}")
    data = pd.DataFrame(
      {
        "S" : np.hstack([self.entropy[0].ravel(), self.entropy[1].ravel()]),
        "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
        "c" : ["node_hierarchy"] * 2 * dim + ["node_hierarchy_H"] * 2 * dim,
        "level" : list(np.arange(dim, 0, -1)) * 4
      }
    )
    dim = self.entropy[2].shape[1]
    print(f"Levels link hierarchy: {dim}")
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            "S" : np.hstack([self.entropy[2].ravel(), self.entropy[3].ravel()]),
            "dir" : ["H"] * dim + ["V"] * dim + ["H"] * dim + ["V"] * dim,
            "c" : ["link_hierarchy"] * 2 * dim + ["link_hierarchy_H"] * 2 * dim,
            "level" : list(np.arange(dim, 0, -1)) * 4
          }
        )
      ], ignore_index=True
    )
    mx = data.iloc[
      data.groupby(["c", "dir"])["S"].transform("idxmax").drop_duplicates().to_numpy()
    ].sort_values("c", ascending=False)
    print(mx)
    # Create figure ----
    g = sns.FacetGrid(
      data=data,
      col = "c",
      hue = "dir",
      col_wrap=2,
      sharex=False,
      sharey=False
    )
    g.map_dataframe(
      sns.lineplot,
      x="level",
      y="S"
    )#.set(xscale="log")
    g.add_legend()
    fig = plt.gcf()
    fig.set_figheight(height)
    fig.set_figwidth(width)

  def plotD(self, width=8, height=6, **kwargs):
    print("\t> Plot D as a function of K")
    # Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    sns.lineplot(
      data=self.FH,
      x="K",
      y="D",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")

  def plotX(self, **kwargs):
    print("Plot X as a function of K")
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(
      data=self.FH,
      x="K",
      y="X",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")

  def plotSloop(self, width=8, height=6, **kwargs):
    print("\t> Plot loop entropy as a function of K")
    # Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    sns.lineplot(
      data=self.FH,
      x="K",
      y="S",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")

  def plotSD(self, width=8, height=6, **kwargs):
    print("Plot SD as a function of K")
    # Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    sns.lineplot(
      data=self.FH,
      x="K",
      y="SD",
      ax=ax
    )
    plt.legend([],[], frameon=False)
    plt.xscale("log")
  
  def nodal_dendrogram(self, r : int, cmap_name="hls", remove_labels=False, width=10, height=7, **kwargs):
    print("Visualize nodal dendrogram!!!")
    from scipy.cluster import hierarchy
    import matplotlib.colors
    # Create figure ----
    if r == 1: r += 1
    partition = hierarchy.cut_tree(self.Z, r).ravel()
    new_partition = skim_partition(partition)
    unique_clusters_id = np.unique(new_partition)
    cm = sns.color_palette(cmap_name, len(unique_clusters_id))
    dlf_col = "#808080"
    ##
    D_leaf_colors = {}
    for i, _ in enumerate(self.colregion.labels[:self.nodes]):
      if new_partition[i] != -1:
        D_leaf_colors[i] = matplotlib.colors.to_hex(cm[new_partition[i]])
      else: D_leaf_colors[i] = dlf_col
    ##
    link_cols = {}
    for i, i12 in enumerate(self.Z[:,:2].astype(int)):
      c1, c2 = (link_cols[x] if x > len(self.Z) else D_leaf_colors[x]
        for x in i12)
      link_cols[i+1+len(self.Z)] = c1 if c1 == c2 else dlf_col
    fig, _ = plt.subplots(1, 1)
    if not remove_labels:
      hierarchy.dendrogram(
        self.Z,
        labels=self.colregion.labels[:self.nodes],
        color_threshold=self.Z[self.nodes - r, 2],
        link_color_func = lambda k: link_cols[k], **kwargs
      )
    else:
      hierarchy.dendrogram(
        self.Z,
        no_labels=True,
        color_threshold=self.Z[self.nodes - r, 2],
        link_color_func = lambda k: link_cols[k], **kwargs
      )
    fig.set_figwidth(width)
    fig.set_figheight(height)

  def heatmap_size(self, width=10, height=10, remove_labels=False, **kwargs):
    print("Visualize network heatmap ordered by nodal community sizes!!!")
    if "labels" in kwargs.keys():
      ids = kwargs["labels"]
      I, fq = sort_by_size(ids, self.nodes)
    else:
      I = np.arange(self.nodes, dtype=int)
      fq = {}
    # Transform FLNs ----
    W = self.R.copy()
    W[~self.nonzero] = np.nan
    W = W[I, :][:, I]
    # Configure labels ----
    labels = self.colregion.labels[I]
    rlabels = [
      str(r) for r in self.colregion.regions[
        "AREA"
      ]
    ]
    colors = self.colregion.regions.loc[
      match(
        labels,
        rlabels
      ),
      "COLOR"
    ].to_numpy()
    # Create figure ----
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(width)
    fig.set_figheight(height)
    if not remove_labels:
      plot = sns.heatmap(
        W,
        cmap=sns.color_palette("viridis", as_cmap=True),
        xticklabels=labels[:self.nodes],
        yticklabels=labels,
        ax = ax
      )
      # Font size ----
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          plot.set_xticklabels(
            plot.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          plot.set_yticklabels(
            plot.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
    else:
      sns.heatmap(
        W,
        cmap=sns.color_palette("viridis", as_cmap=True),
        xticklabels=False,
        yticklabels=False,
        ax = ax
      )
    # Setting labels colors ----
    [t.set_color(i) for i,t in
      zip(
        colors,
        ax.xaxis.get_ticklabels()
      )
    ]
    [t.set_color(i) for i,t in
      zip(
        colors,
        ax.yaxis.get_ticklabels()
      )
    ]
    # Add black lines ----
    if "labels" in kwargs.keys():
      c = 0
      for key in fq:
        c += fq[key]
        if c < self.nodes:
          ax.vlines(
            c, ymin=0, ymax=self.nodes,
            linewidth=2,
            colors=["#C70039"]
          )
          ax.hlines(
            c, xmin=0, xmax=self.nodes,
            linewidth=2,
            colors=["#C70039"]
          )

  def heatmap_dendro(self, r, R, width=8, height=6, linewidth=1, **kwargs):
    W = R.copy()
    W[W == 0] = np.nan
    W[W == -np.Inf] = np.nan
    # Get nodes ordering ----
    from scipy.cluster import hierarchy
    den_order = np.array(
      hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]
    ).astype(int)
    memberships = hierarchy.cut_tree(self.Z, r).ravel()
    memberships = skim_partition(memberships)[den_order]
    C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
    D = np.where(memberships == -1)[0] + 1
    C = list(set(C).union(set(list(D))))
    W = W[den_order, :][:, den_order]
    # Configure labels ----
    labels = self.colregion.labels
    labels =  np.char.lower(labels[den_order].astype(str))
    rlabels = [
      str(re) for re in self.colregion.regions[
        "AREA"
      ]
    ]
    colors = self.colregion.regions.loc[
      match(
        labels,
        rlabels
      ),
      "COLOR"
    ].to_numpy()
    # Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    sns.heatmap(
      W,
      cmap="viridis",
      xticklabels=labels,
      yticklabels=labels,
      ax = ax
    )
    if "font_size" in kwargs.keys():
      if kwargs["font_size"] > 0:
        ax.set_xticklabels(
          ax.get_xmajorticklabels(), fontsize = kwargs["font_size"]
        )
        ax.set_yticklabels(
          ax.get_ymajorticklabels(), fontsize = kwargs["font_size"]
        )
    for c in C:
      ax.vlines(
        c, ymin=0, ymax=self.nodes,
        linewidth=linewidth,
        colors=["#C70039"]
      )
      ax.hlines(
        c, xmin=0, xmax=self.nodes,
        linewidth=linewidth,
        colors=["#C70039"]
      )
    # Setting labels colors ----
    [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
    [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]
    plt.xticks(rotation=90) 

  def lcmap_size(
    self, K, cmap_name="husl", width=10, height=10, remove_labels=False, undirected=False, **kwargs
  ):
    print("Visualize LC memberships ordered by nodal community sizes!!!")
    # Get labels ----
    labels = self.colregion.labels
    regions = self.colregion.regions
    if "labels" in kwargs.keys():
      ids = kwargs["labels"]
      I, fq = sort_by_size(ids, self.nodes)
      flag_fq = True
    else:
      I = np.arange(self.nodes, dtype=int)
      fq = {}
      flag_fq = False
    if "order" in kwargs.keys():
      I = kwargs["order"]
    for k in K:
      # FLN to dataframe and filter FLN = 0 ----
      dA = self.dA.copy()
      # Add id with aesthethis ----
      from scipy.cluster.hierarchy import cut_tree
      if not undirected:
          dA["id"] =  cut_tree(
            self.H,
            n_clusters = k
          ).ravel()
      else:
        dA["id"] = np.tile(cut_tree(
          self.H,
          n_clusters = k
        ).ravel(), 2)
      minus_one_Dc(dA, undirected)
      aesthetic_ids(dA)
      keff = np.unique(
        dA["id"].to_numpy()
      ).shape[0]

      dA = df2adj(dA, var="id")
      dA = dA[I, :][:, I]
      dA[dA == 0] = np.nan
      dA[dA > 0] = dA[dA > 0] - 1

      # Configure labels ----
      labels =  np.char.lower(labels[I].astype(str))
      rlabels = np.array([str(r).lower() for r in regions.AREA])
      colors = regions.COLOR.loc[match(labels, rlabels)].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1, figsize=(width, height))
      # Check colors with and without trees (-1) ---
      if -1 in dA:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        cmap_heatmap = sns.color_palette(cmap_name, keff)
      if not remove_labels:
        plot = sns.heatmap(
          dA,
          xticklabels=labels[:self.nodes],
          yticklabels=labels,
          cmap=cmap_heatmap,
          ax = ax
        )
        if "font_size" in kwargs.keys():
          if kwargs["font_size"] > 0:
            plot.set_xticklabels(
              plot.get_xmajorticklabels(), fontsize = kwargs["font_size"]
            )
            plot.set_yticklabels(
              plot.get_ymajorticklabels(), fontsize = kwargs["font_size"]
            )
        # Setting labels colors ----
        [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
        [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]
      else:
        sns.heatmap(
          dA,
          xticklabels=False,
          yticklabels=False,
          cmap=cmap_heatmap,
          ax = ax
        )
      # Add black lines ----
      if flag_fq:
        c = 0
        for key in fq:
          c += fq[key]
          if c < self.nodes:
            ax.vlines(
              c, ymin=0, ymax=self.nodes,
              colors=["black"]
            )
            ax.hlines(
              c, xmin=0, xmax=self.nodes,
              colors=["black"]
            )
    
  def lcmap_dendro(
    self, R : int, K : int, cmap_name="hls", remove_labels= False,
    width=18, height=15, linewidth=1, undirected=False, **kwargs
  ):
    # Get labels ----
    labels = self.colregion.labels
    regions = self.colregion.regions
    dA = self.dA.copy()
    # Add id with aesthethis ----
    from scipy.cluster.hierarchy import cut_tree
    if not undirected:
      dA["id"] =  cut_tree(
        self.H,
        n_clusters = K
      ).ravel()
    else:
      dA["id"] =  np.tile(cut_tree(
        self.H,
        n_clusters = K
      ).ravel(), 2)

    dA["source_label"] = labels[dA.source]
    dA["target_label"] = labels[dA.target]

    minus_one_Dc(dA, undirected)
    aesthetic_ids(dA)
    keff = np.unique(dA.id)
    keff = keff.shape[0]
    # Transform dFLN to Adj ----
    dA = df2adj(dA, var="id")
    # Get nodes ordering ----
    from scipy.cluster import hierarchy
    den_order = np.array(hierarchy.dendrogram(self.Z, no_plot=True)["ivl"]).astype(int)
    memberships = hierarchy.cut_tree(self.Z, R).ravel()
    memberships = skim_partition(memberships)[den_order]
    C = [i+1 for i in np.arange(len(memberships)-1) if memberships[i] != memberships[i+1]]
    D = np.where(memberships == -1)[0] + 1
    C = list(set(C).union(set(list(D))))
    #
    dA = dA[den_order, :]
    dA = dA[:, den_order]
    dA[dA == 0] = np.nan
    dA[dA > 0] = dA[dA > 0] - 1
    # Configure labels ----
    labels =  np.char.lower(labels[den_order].astype(str))
    rlabels = np.array([str(r).lower() for r in regions.AREA])
    colors = regions.COLOR.loc[match( labels, rlabels)].to_numpy()
    # Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    # Check colors with and without trees (-1) ---
    if -1 in dA:
      save_colors = sns.color_palette(cmap_name, keff - 1)
      cmap_heatmap = [[]] * keff
      cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
      cmap_heatmap[1:] = save_colors
    else:
      cmap_heatmap = sns.color_palette(cmap_name, keff)
    if not remove_labels:
      plot = sns.heatmap(
        dA,
        cmap=cmap_heatmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
      )
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          plot.set_xticklabels(
            plot.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          plot.set_yticklabels(
            plot.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
      # Setting labels colors ----
      [t.set_color(i) for i,t in zip(colors, ax.xaxis.get_ticklabels())]
      [t.set_color(i) for i,t in zip(colors, ax.yaxis.get_ticklabels())]
    else:
      sns.heatmap(
        dA,
        cmap=cmap_heatmap,
        xticklabels=False,
        yticklabels=False,
        ax=ax
      )
    for c in C:
      ax.vlines(
        c, ymin=0, ymax=self.nodes,
        linewidth=linewidth,
        colors=["black"]
      )
      ax.hlines(
        c, xmin=0, xmax=self.nodes,
        linewidth=linewidth,
        colors=["black"]
      )
    plt.xticks(rotation=90)

  def plot_network_simple(self, rlabels, cmap_name="husl", width=10, height=10, **kwargs):
    print("\t> Plot network!!!")
    rlabels = skim_partition(rlabels)
    unique_labels = np.unique(rlabels)
    number_of_communities = unique_labels.shape[0]
    if -1 in unique_labels:
      save_colors = sns.color_palette(cmap_name, number_of_communities - 1)
      color_map = [[]] * number_of_communities
      color_map[0] = [199/ 255.0, 0, 57/ 255.0]
      color_map[1:] = save_colors
    else:
      color_map = sns.color_palette(cmap_name, number_of_communities)
    color_dict = dict()
    for i, lab in enumerate(unique_labels):
      if lab != -1: color_dict[lab] = color_map[i]
      else: color_dict[-1] = "#808080"
    node_colors = [
      color_dict[lab] for lab in rlabels
    ]
    G = nx.from_numpy_array(
      self.A, create_using=nx.DiGraph
    )
    Ainv = self.A.copy()
    Ainv[Ainv != 0] = 1 / Ainv[Ainv != 0]
    Ginv = nx.from_numpy_array(
      Ainv, create_using=nx.DiGraph
    )
    # Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    pos = nx.kamada_kawai_layout(Ginv)
    pos = nx.spring_layout(
      G, pos=pos, iterations=5, seed=212
    )
    nx.draw_networkx(
      G,
      pos=pos,
      node_color=node_colors,
      connectionstyle="arc3,rad=-0.2",
      ax=ax, **kwargs
    )
  
  def plot_link_communities(self, K : int, cmap_name="hls", width=10, height=10, undirected=False, **kwargs):
    print("Draw networkx link communities!!!")
    dA = self.dA.copy()
    from scipy.cluster.hierarchy import cut_tree
    labels = cut_tree(self.H, K).ravel()
    dA["id"] = labels
    minus_one_Dc(dA, undirected=undirected)
    aesthetic_ids(dA)
    labels = dA.id.to_numpy()
    labels[labels > 0] = labels[labels > 0] - 1
    unique_labels = np.unique(labels)
    number_of_communities = unique_labels.shape[0]
    if -1 in unique_labels:
      save_colors = sns.color_palette(cmap_name, number_of_communities - 1)
      color_map = [[]] * number_of_communities
      color_map[0] = [199/ 255.0, 0, 57/ 255.0]
      color_map[1:] = save_colors
    else:
      color_map = sns.color_palette(cmap_name, number_of_communities)
    color_dict = dict()
    for i, lab in enumerate(unique_labels):
      if lab != -1: color_dict[lab] = color_map[i]
      else: color_dict[-1] = "#808080"
    edge_colors = [
      color_dict[lab] for lab in labels
    ]
    G = nx.from_numpy_array(
      self.A, create_using=nx.DiGraph
    )
    Ainv = self.A.copy()
    Ainv[Ainv != 0] = 1 / Ainv[Ainv != 0]
    Ginv = nx.from_numpy_array(
      Ainv, create_using=nx.DiGraph
    )
    # Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    pos = nx.kamada_kawai_layout(Ginv)
    pos = nx.spring_layout(
      G, pos=pos, iterations=5, seed=212
    )
    nx.draw_networkx(
      G, pos=pos,
      edge_color=edge_colors,
      connectionstyle="arc3,rad=-0.2",
      ax=ax, **kwargs
    )

  def plot_network_combined(self, k, R, partition, nocs : dict, sizes : dict, ang=0, width=8, height=8, cmap_name="hls", font_size=10, undirected=False, **kwargs):
      from scipy.cluster import hierarchy
      # Skim partition ----
      new_partition = skim_partition(partition)
      unique_clusters_id = np.unique(new_partition)
      keff = len(unique_clusters_id)
      # Generate all the colors in the color map -----
      if -1 in unique_clusters_id:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        save_colors = sns.color_palette(cmap_name, keff)
        cmap_heatmap = [[]] * (keff+1)
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      # Assign memberships to nodes ----
      if -1 in unique_clusters_id:
        nodes_memberships = {
          k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(len(partition))
        }
      else:
        nodes_memberships = {
          k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(len(partition))
        }
      for i, id in enumerate(new_partition):
        if id == -1: continue
        nodes_memberships[i]["id"][id + 1] = 1
        nodes_memberships[i]["size"][id + 1] = 1
      for i, key in enumerate(nocs.keys()):
        index_key = np.where(self.labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
      # Check unassigned ----
      for i in np.arange(len(partition)):
        if np.sum(nodes_memberships[i] == 0) == keff:
          nodes_memberships[i]["id"][0] = 1
          nodes_memberships[i]["size"][0] = 1
      # Get edges colors ----
      dA = self.dA.copy()
      if not undirected:
        dA["id"] = hierarchy.cut_tree(self.H, k).reshape(-1)
      else:
         dA["id"] = np.tile(hierarchy.cut_tree(self.H, k).reshape(-1), 2)
      minus_one_Dc(dA, undirected)
      aesthetic_ids(dA)
      dA = df2adj(dA, var="id")
      # Generate graph ----
      G = nx.DiGraph(R)
      r_min = np.min(R[R>0])
      r_max = np.max(R)
      edge_color = [""] * self.leaves
      for i, dat in enumerate(G.edges(data=True)):
        u, v, a = dat
        if "coords" not in kwargs.keys():
          G[u][v]["kk_weight"] = - (a["weight"] - r_min) / (r_max - r_min) + r_max
        if dA[u, v] == -1: edge_color[i] = cmap_heatmap[0]
        else: edge_color[i] = "gray"
      if "coords" not in kwargs.keys():
        pos = nx.kamada_kawai_layout(G, weight="kk_weight")
      else:
        pos = kwargs["coords"]
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      labs = {k : lab for k, lab in zip(G.nodes, self.labels)}
      plt.figure(figsize=(width, height))
      if "not_edges" not in kwargs.keys():
        nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color, alpha=0.2, arrowsize=20, connectionstyle="arc3,rad=-0.1")
      if "modified_labels" not in kwargs.keys():
        nx.draw_networkx_labels(G, pos=pos, labels=labs, font_size=font_size)
      else:
        nx.draw_networkx_labels(G, pos=pos, labels=kwargs["modified_labels"], font_size=font_size)
      for node in G.nodes:
        a = plt.pie(
          [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
          center=pos[node], 
          colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
          radius=0.05
        )
      array_pos = np.array([list(pos[v]) for v in pos.keys()])
      plt.xlim(-0.1 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.1)
      plt.ylim(-0.1 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.1)

  def plot_network_kk(self, R, partition, nocs : dict, sizes : dict, ang=0, width=8, height=6, front_edges=False, font_size=0.1, undirected=False, cmap_name="hls"):
      new_partition = skim_partition(partition)
      unique_clusters_id = np.unique(new_partition)
      keff = len(unique_clusters_id)
      save_colors = sns.color_palette(cmap_name, keff - 1)
      cmap_heatmap = [[]] * keff
      cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
      cmap_heatmap[1:] = save_colors
      # Assign memberships to nodes ----
      if -1 in unique_clusters_id:
        nodes_memberships = {k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(self.nodes)}
      else:
        nodes_memberships = {k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(self.nodes)}
      for i, id in enumerate(new_partition):
        if id == -1: continue
        nodes_memberships[i]["id"][id + 1] = 1
        nodes_memberships[i]["size"][id + 1] = 1
      for i, key in enumerate(nocs.keys()):
        index_key = np.where(self.labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
      # Check unassigned ----
      for i in np.arange(self.nodes):
        if np.sum(np.array(nodes_memberships[i]["id"]) == 0) == keff:
          nodes_memberships[i]["id"][0] = 1
          nodes_memberships[i]["size"][0] = 1
        # elif np.sum(np.array(nodes_memberships[i]) != 0) > 2:
        #   print(nodes_memberships[i])
      if not undirected:
        G = nx.DiGraph(R)
      else:
        G = nx.Graph(R, directed=False)
      pos = nx.kamada_kawai_layout(G)
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      fig, ax = plt.subplots(1, 1, figsize=(width, height))
      # Create labels ---
      labs = {n : self.labels[n] for n in G.nodes}
      nx.draw_networkx_labels(G, pos=pos, labels=labs, font_size=font_size, ax=ax)
      if not front_edges:
        if undirected:
          nx.draw_networkx_edges(G, pos=pos, arrows=False, ax=ax)
        else:
          nx.draw_networkx_edges(G, pos=pos, arrows=True, ax=ax)
      for node in G.nodes:

        plt.pie(
          [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
          center=pos[node], 
          colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
          radius=0.05
        )
      if front_edges:
        if undirected:
          nx.draw_networkx_edges(G, pos=pos, arrows=False, ax=ax)
        else:
          nx.draw_networkx_edges(G, pos=pos, arrows=True, ax=ax)
      array_pos = np.array([list(pos[v]) for v in pos.keys()])
      plt.xlim(-0.1 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.1)
      plt.ylim(-0.1 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.1)
