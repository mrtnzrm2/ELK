# Standard libs ----
import numpy as np
import os
# Personal libs ----
from various.network_tools import *

class BASE:
  def __init__(self, linkage, **kwargs) -> None:
    # Set general attributes ----
    self.linkage = linkage
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = ""
    if "version" in kwargs.keys():
      self.version = str(kwargs["version"])
    else: self.version = ""
    if "mode" in kwargs.keys():
      self.mode = kwargs["mode"]
    else: self.mode = "ALPHA"
    if "topology" in kwargs.keys():
      self.topology = kwargs["topology"]
    else: self.topology = ""
    # Save methos from net_tool ----
    self.column_normalize = column_normalize
    self.save_class = save_class
    self.read_class = read_class
    # Create paths ----
    self.common_path = os.path.join(
      self.structure, self.version
    )