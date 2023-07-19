# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
# Personal libs ----
from modules.hieraranalysis import Hierarchy
from various.network_tools import *

class Plot_H:
  def __init__(self, plot_path, H : Hierarchy) -> None:
    ## Attributes ----
    self.linkage = H.linkage
    self.BH = H.FH
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

    # Net ----
    self.path = plot_path
    self.areas = H.colregion.labels

    # Get regions and colors ----
    self.colregion = H.colregion
    self.colregion.get_regions()
    
  def plot_newick_R(self, tree_newick, weighted=False, on=True):
    if on:
      print("\t> Plot tree in Newick format from R!!!")
      import subprocess
      # Arrange path ----
      plot_path = join(self.path, "NEWICK")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      if not weighted:
        subprocess.run(["Rscript", "R/plot_newick_tree.R", tree_newick, join(plot_path, "tree_newick.png")])
      else:
        subprocess.run(["Rscript", "R/plot_newick_tree_H.R", tree_newick, join(plot_path, "tree_newick_H.png")])

  def plotD(self, on=False, **kwargs):
    if on:
      print("\t> Plot D as function of K")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="D",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "D_logK.png"
        ),
        dpi=300
      )
      plt.close()

  def plotSloop(self, on=False, **kwargs):
    if on:
      print("\t> Plot loop entropy as a function of K")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="S",
        ax=ax
      )
      ax.set_ylabel(r"$H_{L}$")
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "S_logK.png"
        ),
        dpi=300
      )
      plt.close()

  def plotSD(self, on=False, **kwargs):
    if on:
      print("Plot SD as a function of K")
      # Create figure ----
      if "SD" not in self.BH[0].columns:
        self.BH[0]["SD"] = (self.BH[0].D / np.nansum(self.BH[0].D)) * (self.BH[0].S / np.nansum(self.BH[0].S))
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="SD",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "SD_logK.png"
        ),
        dpi=300
      )
      plt.close()

  def plotX(self, on=False, **kwargs):
    if on:
      print("Plot X as a function of K")
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=self.BH[0],
        x="K",
        y="X",
        ax=ax
      )
      plt.legend([],[], frameon=False)
      plt.xscale("log")
      fig.tight_layout()
      # Arrange path ----
      plot_path = join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "X_logK.png"
        ),
        dpi=300
      )
      plt.close()
  
  def nodal_dendrogram(self, R : list, score="", cmap_name="hls", remove_labels=False, on=False, **kwargs):
    if on:
      print("\t> Visualize nodal dendrogram!!!")
      # Arrange path ----
      plot_path = os.path.join(self.path, "Features")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      from scipy.cluster import hierarchy
      import matplotlib.colors
      # Create figure ----
      for r in R:
        if r == 1:
          r += 1
          sname = "fake"
        else: sname = ""
        #
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
            link_color_func = lambda k: link_cols[k],
            leaf_rotation=90, **kwargs
          )
        else:
          hierarchy.dendrogram(
            self.Z,
            no_labels=True,
            color_threshold=self.Z[self.nodes - r, 2],
            link_color_func = lambda k: link_cols[k]
          )
        fig.set_figwidth(10)
        fig.set_figheight(7)
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "core_dendrogram_{}_{}{}{}.png".format(self.linkage, r, score, sname)
          ),
          dpi=500
        )
        plt.close()

  def heatmap_size(self, r, R, score="", linewidth=1.5, on=True, **kwargs):
    if on:
      print("\t> Visualize R heatmap ordered by nodal community size!!!")
      if "labels" in kwargs.keys():
        ids = kwargs["labels"]
        I, fq = sort_by_size(ids, self.nodes)
      else:
        I = np.arange(self.nodes, dtype=int)
        fq = {}

      W = R.copy()
      nonzero = (W != 0)
      W[~nonzero] = np.nan
      W = W[I, :][:, I]
      # Configure labels ----
      labels = self.colregion.labels[I]
      labels = [str(r) for r in labels]
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
      fig.set_figwidth(19)
      fig.set_figheight(15 * W.shape[0]/ self.nodes)
      sns.heatmap(
        W,
        cmap=sns.color_palette("viridis", as_cmap=True),
        xticklabels=labels[:self.nodes],
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
              linewidth=linewidth,
              colors=["#C70039"]
            )
            ax.hlines(
              c, xmin=0, xmax=self.nodes,
              linewidth=linewidth,
              colors=["#C70039"]
            )
     # Arrange path ----
      plot_path = os.path.join(self.path, "Heatmap_size")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "r_{}_{}.png".format(r, score)
        ),
        dpi = 300
      )
      plt.close()

  def heatmap_dendro(self, r, R, score="", linewidth=1.5, on=True, **kwargs):
    if on:
      print("\t> Visualize R heatmap ordered by the node dendrogram!!!")

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
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(18)
      fig.set_figheight(15)
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
      plt.yticks(rotation=0)
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Heatmap_dendrogram"
      )
      # Crate path ----
      Path(
        plot_path
    ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, f"r_{r}_{score}.png"
        ),
        dpi = 300
      )
      plt.close()

  def lcmap_size(self, K, score="", cmap_name="husl", linewidth=1.5, undirected=False, on=True, **kwargs):
    if on:
      print("\t> Visualize LC memberships ordering by nodal community sizes!!!")
      # Get elemets from colregion ----
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
        # Transform dFLN to Adj ----
        dA = df2adj(dA, var="id")
        dA = dA[I, :][:, I]
        dA[dA == 0] = np.nan
        dA[dA > 0] = dA[dA > 0] - 1

        # Configure labels ----
        labels = self.colregion.labels[I]
        labels = [str(r) for r in labels]
        rlabels = [str(r) for r in regions.AREA]
        colors = regions.COLOR.loc[match(labels, rlabels)].to_numpy()
        # Create figure ----
        fig, ax = plt.subplots(1, 1)
        # Check colors with and without trees (-1) ---
        if -1 in dA:
          save_colors = sns.color_palette(cmap_name, keff - 1)
          cmap_heatmap = [[]] * keff
          cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
          cmap_heatmap[1:] = save_colors
        else:
          cmap_heatmap = sns.color_palette(cmap_name, keff)
        fig.set_figwidth(19)
        fig.set_figheight(15 * dA.shape[0]/ self.nodes)
        sns.heatmap(
          dA,
          xticklabels=labels[:self.nodes],
          yticklabels=labels,
          cmap=cmap_heatmap,
          annot_kws = {"size" : 12},
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
        if flag_fq:
          c = 0
          for key in fq:
            c += fq[key]
            if c < self.nodes:
              ax.vlines(
                c, ymin=0, ymax=self.nodes,
                colors=["black"], linewidth = linewidth
              )
              ax.hlines(
                c, xmin=0, xmax=self.nodes,
                colors=["black"], linewidth = linewidth
              )
        # Arrange path ----
        plot_path = os.path.join(self.path, "Link_memberships_sizes")
        # Crate path ----
        Path(
          plot_path
        ).mkdir(exist_ok=True, parents=True)
        # Save plot ----
        plt.savefig(
          os.path.join(
            plot_path, "k_{}_{}.png".format(k, score)
          ),
          dpi = 300
        )
        plt.close()

  def lcmap_dendro(
    self, K, R, score="", cmap_name="hls", remove_labels=False, linewidth=1.5, undirected=False, on=False, **kwargs
  ):
    if on:
      print("\t> Visualize LC memberships ordered by the node hierarchy!!!")
      # K loop ----
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Link_memberships_dendrogram"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Get labels ----
      labels = self.colregion.labels
      regions = self.colregion.regions
      # FLN to dataframe and filter FLN = 0 ----
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
      ##
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
      fig, ax = plt.subplots(1, 1)
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
      fig.set_figwidth(18)
      fig.set_figheight(15)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "k_{}_{}.png".format(K, score)
        )
      )
      plt.close()