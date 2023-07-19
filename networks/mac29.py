from os.path import join
import pandas as pd
import numpy as np
# ELK libraries ----
from networks.base import BASE
from various.network_tools import *

class MAC29(BASE):
  def __init__(
    self, **kwargs
  ) -> None:
    super().__init__(subject="MAC", structure="ANLNe", version="29d91", **kwargs)
     #  Paths as labels
    self.csv_path = "../CSV/MAC/29d91/"
    self.distance_path = f"../CSV/MAC/29d91/MAP3D/"
    self.labels_path = self.csv_path
    self.regions_path = join(
      "../CSV/Regions",
      "Table_areas_regions_09_2019.csv"
    )
    # Get structure network ----
    self.C, self.A = self.get_structure()
    # Get network's spatial distances ----
    self.D = self.get_distance_MAP3D()
    # Get regions ----
    self.get_regions()

  def get_structure(self):
    # Get structure ----
    file = pd.read_csv(f"{self.csv_path}/Neurons91x29_Arithmean_DBV23.45.csv", index_col=0, header=0)
    col_labs = list(file.columns)
    row_labs = list(file.index)
    new_labs = col_labs + [r for r in row_labs if r not in col_labs]
    file = file.loc[new_labs, :]
    C = file.to_numpy()
    A = C / np.sum(C, axis=0)
    self.rows = C.shape[0]
    self.nodes = C.shape[1]
    self.struct_labels = new_labs
    self.struct_labels = np.char.lower(self.struct_labels)
    self.labels = self.struct_labels
    return C.astype(float), A.astype(float)

  def get_distance_MAP3D(self):
    fname =  join(self.distance_path, "DistanceMatrix Map3Dmars2019_91x91.csv")
    file = pd.read_csv(fname, index_col=0)
    clabel =file.columns.to_numpy()
    clabel = np.array([str(lab) for lab in clabel])
    clabel = np.char.lower(clabel)

    D2C = {
      "pi" : "parainsula"
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
      
