# Python libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
# ELK libs ----
from various.network_tools import *

class Plot_N:
  def __init__(self) -> None:
    pass

  def histogramX(self, X, width=8, height=6):
    dA = adj2df(X)
    dA = dA.loc[dA.weight != 0]
    fig , ax = plt.subplots(1, 1, figsize=(width, height))
    sns.histplot(
      data=dA,
      x="weight",
      stat="density",
      ax=ax
    )
    fig.tight_layout()
  
  def regXY(self, X, Y, s=1, lowess=False, width=8, height=6):
    # Get data ----
    dY = adj2df(Y.copy())
    dX = adj2df(X.copy())
    # Get No connections ---
    yzeros = dY.weight == 0
    yisnan = np.isnan(dY.weight)
    yinf = (dY.weight == np.Inf) | (dY.weight == -np.Inf)
    yno = (~yzeros) & (~yisnan) & (~yinf)

    xzeros = dX.weight == 0
    xisnan = np.isnan(dX.weight)
    xinf = (dX.weight == np.Inf) | (dX.weight == -np.Inf)
    xno = (~xzeros) & (~xisnan) & (~xinf)

    no = yno | xno
    # Eliminate zeros ---
    dY = dY.loc[no]
    dX = dX.loc[no]
    # Create data ----
    data = pd.DataFrame(
      {
        "Y" : dY["weight"],
        "X" : dX["weight"]
      }
    )
    # Create figure ----
    _ , ax = plt.subplots(1, 1, figsize=(width, height))
    sns.regplot(
      data=data,
      x="X",
      y="Y",
      lowess=lowess,
      scatter_kws={"s" : s},
      line_kws={"color" : "orange"},
      ax=ax
    )

  def scatterXY(self, X, Y, s=1, width=8, height=7):
    # Get data ----
    dY = adj2df(Y.copy())
    dX = adj2df(X.copy())
    # Get No connections ---
    zeros = dY.weight == 0
    isnan = np.isnan(dY.weight)
    # Eliminate zeros ---
    dY = dY.loc[(~zeros) & (~isnan)]
    dX = dX.loc[(~zeros) & (~isnan)]
    # Create data ----
    data = pd.DataFrame(
      {
        "Y" : dY["weight"],
        "X" : dX["weight"]
      }
    )
    # Create figure ----
    _ , ax = plt.subplots(1, 1, figsize=(width, height))
    sns.scatterplot(
      data=data,
      x="X",
      y="Y",
      s=s,
      ax=ax
    )
