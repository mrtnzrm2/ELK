# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
# Personal libs ----
from various.network_tools import *
from modules.hieraranalysis import Hierarchy

class Plot_N:
  def __init__(self, plot_path, H : Hierarchy) -> None:
    # From net ----
    self.path = plot_path
    # from Hierarchy ----
    self.nodes = H.nodes
    self.edges = H.leaves
    self.linkage = H.linkage
    self.A = H.A
    self.dA = H.dA
    self.FH = H.FH
    self.index = H.index
    self.H = H.H

  def histogramX(self, A, label="", on=True):
    if on:
      print("Plot weight histogram!!!")
      # Transform FLN to DataFrame ----
      dA = adj2df(A)
      dA["connection"] = "exist"
      dA.connection.loc[dA.weight == 0] = "~exist"
      # Transform FLN to weights ----
      fig, ax = plt.subplots(1, 1)

      sns.histplot(
        data=dA.loc[dA.connection == "exist"],
        x="weight",
        hue = "connection",
        stat="density",
        ax=ax
      )
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(
        self.path,"Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, f"R_histogram_{label}.png"
        ),
        dpi=300
      )
      plt.close()

  def plot_network_kk(self, H : Hierarchy, partition, nocs : dict, sizes : dict, labels, ang=0, score="", front_edges=False, log_data=False, font_size=0.1, undirected=False, cmap_name="hls", on=True):
    if on:
      print("\t> Printing network using the kammada-kawai layout")
      new_partition = skim_partition(partition)
      unique_clusters_id = np.unique(new_partition)
      keff = len(unique_clusters_id)
      save_colors = sns.color_palette(cmap_name, keff - 1)
      cmap_heatmap = [[]] * keff
      cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
      cmap_heatmap[1:] = save_colors
      # Assign memberships to nodes ----
      if -1 in unique_clusters_id:
        nodes_memberships = {k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(H.nodes)}
      else:
        nodes_memberships = {k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(H.nodes)}
      for i, id in enumerate(new_partition):
        if id == -1: continue
        nodes_memberships[i]["id"][id + 1] = 1
        nodes_memberships[i]["size"][id + 1] = 1
      for i, key in enumerate(nocs.keys()):
        index_key = np.where(labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
      # Check unassigned ----
      for i in np.arange(H.nodes):
        if np.sum(np.array(nodes_memberships[i]["id"]) == 0) == keff:
          nodes_memberships[i]["id"][0] = 1
          nodes_memberships[i]["size"][0] = 1
        # elif np.sum(np.array(nodes_memberships[i]) != 0) > 2:
        #   print(nodes_memberships[i])
      if not log_data:
        A = H.A
      else:
        A = np.log(1 + H.A)
      if not undirected:
        G = nx.DiGraph(A)
      else:
        G = nx.Graph(A, directed=False)
      pos = nx.kamada_kawai_layout(G)
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      fig, ax = plt.subplots(1, 1)
      # Create labels ---
      labs = {n : labels[n] for n in G.nodes}
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
      fig.set_figheight(9)
      fig.set_figwidth(9)
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Network"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, f"net_kk_{score}.png"
        ),
        dpi=300
      )
  
  def plot_network_covers(self, k, R, partition, nocs : dict, sizes : dict, labels, ang=0, score="", cmap_name="hls", undirected=False, on=True, **kwargs):
    if on:
      print("\t> Printing network covers")
      from scipy.cluster import hierarchy
      # from matplotlib.colors import to_hex
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
        index_key = np.where(labels == key)[0][0]
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
      edge_color = [""] * self.edges
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
      labs = {k : lab for k, lab in zip(G.nodes, labels)}
      plt.figure(figsize=(12,12))
      if "not_edges" not in kwargs.keys():
        nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color, alpha=0.2, arrowsize=20, connectionstyle="arc3,rad=-0.1")
      if "modified_labels" not in kwargs.keys():
        nx.draw_networkx_labels(G, pos=pos, labels=labs)
      else:
        nx.draw_networkx_labels(G, pos=pos, labels=kwargs["modified_labels"])
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
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Network"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, f"net_cover_{score}.png"
        ),
        dpi=300
      )




