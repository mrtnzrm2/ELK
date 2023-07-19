# Python libs ----
import os
# ELK libraries ----
from various.network_tools import *

class BASE:
  def __init__(self, **kwargs) -> None:
    # Set general attributes ----
    if "subject" in kwargs.keys():
      self.subject = kwargs["subject"]
    else: self.subject = ""
    if "structure" in kwargs.keys():
      self.structure = kwargs["structure"]
    else: self.structure = ""
    if "version" in kwargs.keys():
      self.version = str(kwargs["version"])
    else: self.version = ""
    # Save methos from net_tool ----
    self.column_normalize = column_normalize
    self.save_class = save_class
    self.read_class = read_class
    # Create paths ----
    self.common_path = os.path.join(
      self.subject, self.structure, self.version
    )