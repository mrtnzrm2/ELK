import pandas as pd
from matplotlib.colors import to_hex

class colregion:
  def __init__(self, NET) -> None:
    # Define attributes ----
    self.nodes = NET.nodes
    self.labels = NET.labels

  def get_regions(self, add_colregion = (False, False)):
    if not add_colregion[0]:
      self.regions = pd.DataFrame(
        {
          "AREA" : self.labels,
          "REGION" : ["UNDEFINED"] * self.nodes,
          "COLOR" : [to_hex((0., 0., 0.))] * self.nodes
        }
      )
    else:
      if isinstance(add_colregion[1], pd.DataFrame):
        self.regions = add_colregion[1]
      else:
        raise ValueError("Region has to be a panda dataframe.")