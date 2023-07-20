from os.path import join
import os
import pandas as pd
import numpy as np
from pathlib import Path
from various.network_tools import *
from networks.base import BASE

class MAC40(BASE):
  def __init__(
    self, distance="tracto16", **kwargs
  ) -> None:
    super().__init__(**kwargs)
    #  Paths as labels
    self.csv_path = "../CSV/MAC/40d91/"
    self.distance_path = f"../CSV/MAC/40d91/{distance}/"
    self.labels_path = self.csv_path
    self.regions_path = join(
      "../CSV/Regions",
      "Table_areas_regions_09_2019.csv"
    )
    # Get structure network ----
    self.distance = distance
    self.C, self.CC, self.A = self.get_structure()
    # Get network's spatial distances ----
    dist_dic = {
      "MAP3D" : self.get_distance_MAP3D,
      "tracto16" : self.get_distance_tracto16
    }
    self.D = dist_dic[self.distance]()
    self.get_regions()

  def get_structure(self):
    # Get structure ----
    file = pd.read_csv(f"{self.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
    ## Areas to index
    tlabel = np.unique(file.TARGET)
    inj = tlabel.shape[0]
    slabel = np.unique(file.SOURCE)
    total_areas = slabel.shape[0]
    slabel1 = [lab for lab in slabel if lab not in tlabel]
    slabel = np.array(list(tlabel) + slabel1)
    file["SOURCE_IND"] = match(file.SOURCE, slabel)
    file["TARGET_IND"] = match(file.TARGET, slabel)
    ## Average Count
    monkeys = np.unique(file.MONKEY)
    C = []
    tid = np.unique(file.TARGET_IND)
    tmk = {t : [] for t in tid}
    for i, m in enumerate(monkeys):
      Cm = np.zeros((total_areas, inj))
      data_m = file.loc[file.MONKEY == m]
      Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.TOTAL
      C.append(Cm)
      for t in np.unique(data_m.TARGET_IND):
        tmk[t].append(i)
    C = np.array(C)
    C[np.isnan(C)] = 0
    CC = np.sum(C, axis=0)
    c = np.zeros((total_areas, inj))
    for t, mnk in tmk.items(): c[:, t] = np.mean(C[mnk, :, t], axis=0)
    A = CC / np.sum(CC, axis=0)
    self.rows = c.shape[0]
    self.nodes = c.shape[1]
    self.struct_labels = slabel
    self.struct_labels = np.char.lower(self.struct_labels)
    self.labels = self.struct_labels
    # np.savetxt(f"{self.csv_path}/labels.csv", self.struct_labels,  fmt='%s')
    return c.astype(float), CC.astype(float), A.astype(float)

  def get_distance_MAP3D(self):
    fname =  join(self.distance_path, "DistanceMatrix Map3Dmars2019_91x91.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)
    ## Rename areas from D to C
    D2C = {
      "perirhinal" : "peri",
      "entorhinal" : "ento",
      "subiculum" : "sub",
      "temporal_pole" : "pole",
      "insula" : "ins",
      "piriform" : "pir"
    }
    for key, val in D2C.items():
      clabel[clabel == key] = val
    # labs = [lab for lab in clabel if lab not in self.struct_labels]
    # print(labs)
    D = file.to_numpy()
    order = match(self.struct_labels, clabel)
    D = D[order, :][:, order]
    D = np.array(D)
    np.fill_diagonal(D, 0.)
    D = D.astype(float)
    return D
  
  def get_distance_tracto16(self):
    fname =  join(self.distance_path, "Macaque_TractoDist_91x91_220418.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)

    # labs = [lab for lab in clabel if lab not in self.struct_labels]
    # print(labs)
    # ## Rename areas from D to C
    D2C = {
      "subi" : "sub",
      "insula" : "ins"
    }
    for key, val in D2C.items():
      clabel[clabel == key] = val
    D = file.to_numpy()
    order = match(self.struct_labels, clabel)
    D = D[order, :][:, order]
    D = np.array(D)
    np.fill_diagonal(D, 0.)
    D = D.astype(float)
    return D
  
  def MAC_region_colors(self):
    from matplotlib.colors import to_hex
    maxc = 255
    colors = pd.DataFrame(
      {
        "REGION" : [
          "Occipital",
          "Temporal",
          "Parietal",
          "Frontal",
          "Prefrontal",
          "Cingulate"
        ],
        "COLOR" : [
          to_hex((0 ,97/maxc, 65/maxc)),
          to_hex((1, 126/maxc, 0)),
          "#800080",
          "#fec20c",
          # "#ffd500",
          to_hex((237/maxc, 28/maxc, 36/maxc)),
          "#2a52be"
        ]
      }
    )
    return colors
  
  def MAC_areas_regions(self, df):
      df.loc[df["AREA"] == "29-30", "AREA"] = "29/30"
      df.loc[df["AREA"] == "prost", "AREA"] = "pro.st."
      df.loc[df["AREA"] == "tea-m a", "AREA"] = "tea/ma"
      df.loc[df["AREA"] == "tea-m p", "AREA"] = "tea/mp"
      df.loc[df["AREA"] == "th-tf", "AREA"] = "th/tf"
      df.loc[df["AREA"] == "9-46d", "AREA"] = "9/46d"
      df.loc[df["AREA"] == "9-46v", "AREA"] = "9/46v"
      df.loc[df["AREA"] == "opal", "AREA"] = "opai"
      df.loc[df["AREA"] == "t.pole", "AREA"] = "pole"
      df.loc[df["AREA"] == "parains", "AREA"] = "pi"
      df.loc[df["AREA"] == "insula", "AREA"] = "insula"
      df.loc[df["AREA"] == "aud. core", "AREA"] = "core"
      df.loc[df["AREA"] == "35-36", "AREA"] = "35/36"

  def get_regions(self):
      self.regions = pd.read_csv(
        self.regions_path
      )
      self.regions.columns = [
        "AREA", "REGION"
      ]
      self.regions["AREA"] = [
        np.char.lower(x) for x in self.regions["AREA"] if isinstance(x, str)
      ]
      # Format area labels from regions ----
      self.MAC_areas_regions(self.regions)
      # Set colors to region dataframe ----
      colors = self.MAC_region_colors()
      self.regions["COLOR"] = colors.loc[
        match(
          self.regions["REGION"],
          colors["REGION"]
        ),
        "COLOR"
      ].to_numpy()