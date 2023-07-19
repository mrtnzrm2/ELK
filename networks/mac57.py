from os.path import join
import pandas as pd
import numpy as np
from pathlib import Path
from various.network_tools import *
from networks.base import BASE

class MAC57(BASE):
  def __init__(self, distance="tracto16", **kwargs) -> None:
    super().__init__(**kwargs)
    #  Paths as labels
    self.csv_path = "../CSV/MAC/57d106_230605/"
    self.distance_path = f"../CSV/MAC/57d106_230605/{distance}/"
    self.labels_path = self.csv_path
    self.regions_path = join(
      "../CSV/Regions",
      "Table_areas_regions_09_2019.csv"
    )
    # Get structure network ----
    self.C = self.get_structure()
    self.CC, self.A = self.get_summer_counts()
    # Get network's spatial distances ----
    self.D = self.get_distance_tracto16()
    self.get_regions()

  def get_structure(self):
    # Get structure ----
    file = pd.read_csv(f"{self.csv_path}/CountMatrix_Averaged_57areas_230605.csv", index_col=0)
    ## Areas to index
    tlabel = file.columns.to_numpy()
    slabel = file.index.to_numpy()
    slabel1 = [lab for lab in slabel if lab not in tlabel and lab != "Claustrum"]
    slabel = np.array(list(tlabel) + slabel1)
    ## Average Count
    C = file[tlabel].loc[slabel]
    C = C.to_numpy(dtype=float)
    # A = C / np.sum(C, axis=0)
    self.rows = C.shape[0]
    self.nodes = C.shape[1]
    self.struct_labels = slabel
    self.struct_labels = np.char.lower(self.struct_labels)
    self.labels = self.struct_labels
    # np.savetxt(f"{self.csv_path}/labels57.csv", self.struct_labels,  fmt='%s')
    return C.astype(float)

  def get_summer_counts(self):
    file = pd.read_csv(f"{self.csv_path}/CountMatrix_Summed_57areas_220830.csv", index_col=0)
    file.columns = np.char.lower(file.columns.to_numpy(dtype=str))
    file.index = np.char.lower(file.index.to_numpy(dtype=str))
    file = file[self.struct_labels[:self.nodes]].loc[self.struct_labels]
    CC = file.to_numpy(dtype=float)
    A = CC / np.sum(CC, axis=0)
    return CC, A
  
  def get_distance_tracto16(self):
    fname =  join(self.distance_path, "106x106_DistanceMatrix.csv")
    file = pd.read_csv(fname, index_col=0)
    # print(file.columns.to_numpy())
    file.columns = np.char.lower(file.columns.to_numpy(dtype=str))
    file.index = np.char.lower(file.index.to_numpy(dtype=str))
    D = file[self.struct_labels].loc[self.struct_labels]
    D = D.to_numpy()
    np.fill_diagonal(D, 0.)
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
      