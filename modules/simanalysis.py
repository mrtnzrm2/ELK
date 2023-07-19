# Standard libs ----
import numpy as np
# Personal libs ----
import simquest as squest

class Sim:
  def __init__(
    self, nodes : int, A, R, mode, topology="MIX", index="jacp", lookup=0, undirected=False
  ):
    # Parameters ----
    self.nodes = nodes
    self.mode = mode
    self.A = A
    self.R = R
    self.undirected = undirected
    if not self.undirected:
      self.nonzero = (A != 0)
    else:
      self.nonzero = (np.triu(A) != 0)
    # Number of connections in the EC component ----
    if not self.undirected:
      self.leaves = np.sum(self.A[:nodes, :nodes] != 0).astype(int)
    else:
      self.leaves =  int(np.sum(self.A[:nodes, :nodes] != 0) / 2)
    self.topologies = {
      "MIX" : 0, "SOURCE" : 1, "TARGET" : 2
    }
    self.indices = {
      "jacp" : 0, "tanimoto" : 1, "cos" : 2, "bsim" : 3,  "S1_2" : 4
    }
    self.lup = lookup
    self.topology = topology
    self.index = index

  def get_aik(self):
    aik =  self.R.copy()
    aki = self.R.copy().T
    for i in np.arange(self.nodes):
      if self.mode == "ALPHA":
        aik[i, i] = np.nanmean(aik[i, :][aik[i, :] != self.lup])
      elif self.mode == "BETA":
        aik[i, i] = np.nanmean(aki[i, :][aki[i, :] != self.lup])
      elif self.mode == "ZERO":
        aik[i, i] = 0
      elif self.mode == "gALPHA":
        aik[i, i] = np.exp(np.nanmean(np.log(aik[i, :][aik[i, :] != self.lup])))
      elif self.mode == "gBETA":
        aik[i, i] = np.exp(np.nanmean(np.log(aki[i, :][aki[i, :] != self.lup])))
      else:
        raise ValueError("Bad mode")
    return aik

  def get_aki(self):
    aki = self.R.copy().T
    aik = self.R.copy()
    for i in np.arange(self.nodes):
      if self.mode == "ALPHA":
        aki[i, i] = np.nanmean(aki[i, :][aki[i, :] != self.lup])
      elif self.mode == "BETA":
        aki[i, i] = np.nanmean(aik[i, :][aik[i, :] != self.lup])
      elif self.mode == "ZERO":
        aki[i, i] = 0
      elif self.mode == "gALPHA":
        aki[i, i] = np.exp(np.nanmean(np.log(aki[i, :][aki[i, :] != self.lup])))
      elif self.mode == "gBETA":
        aki[i, i] = np.exp(np.nanmean(np.log(aik[i, :][aik[i, :] != self.lup])))
      else:
        raise ValueError("Bad mode")
    return aki

  def get_id_matrix(self):
    self.id_mat = self.A[:self.nodes, :]
    if self.undirected:
      self.id_mat = np.triu(self.id_mat)
    self.id_mat[self.id_mat != 0] = np.arange(1, self.leaves + 1)
    self.id_mat = self.id_mat.astype(int)
  
  def similarity_by_feature_cpp(self):
    Quest = squest.simquest(
      self.nonzero, self.get_aki(), self.get_aik(),
      self.nodes, self.leaves, self.topologies[self.topology],
      self.indices[self.index]
    )
    self.linksim_matrix = np.array(Quest.get_linksim_matrix())
    self.source_sim_matrix = np.array(Quest.get_source_matrix())
    self.target_sim_matrix = np.array(Quest.get_target_matrix())