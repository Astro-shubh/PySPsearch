import argparse
import numpy as np
#import multiprocessing as mp
import os
import sys
import glob
import pandas as pd
from concurrent.futures import wait
from concurrent.futures import ProcessPoolExecutor as Pool
import modules.search_filterbank
import modules.clustering_classes
import modules.plot_files
import matplotlib.pyplot as plt

###########    Argument Parsing     #################

parser = argparse.ArgumentParser(
    description="Single-pulse search and clustering pipeline"
)
parser.add_argument("-f", "--filename", help="Input filterbank filename")
parser.add_argument("-lodm", "--low_dm", help="Smallest DM to search.")
parser.add_argument("-hidm", "--high_dm", help="Largest DM to search.")
parser.add_argument("-mw", "--max_width", help="Maximum width (s)")
parser.add_argument("-th", "--threshold", help="Detection threshold")

# Clustering Selection and Parameters
parser.add_argument(
    "-cm",
    "--cluster_method",
    type=str,
    default="fofW",
    choices=["fofW", "fof", "dbscan", "hdbscan"],
    help="Clustering method to use: fofW, fof, dbscan, hdbscan",
)

parser.add_argument(
    "-mpt",
    "--min_points",
    type=int,
    default=10,
    help="Min points / min_cluster_size for HDBSCAN / DBSCAN",
)

parser.add_argument(
    "--fof_link",
    type=float,
    default=0.05,
    help="Linking length in seconds for fof (s)",
)

parser.add_argument(
    "--eps",
    type=float,
    default=0.05,
    help="Epsilon parameter for DBSCAN max distance (s)",
)

args = parser.parse_args()
max_width = float(args.max_width)
filename = str(args.filename)
threshold = float(args.threshold)
lodm = float(args.low_dm)
hidm = float(args.high_dm)
cluster_method = args.cluster_method.lower()

### Search the filterbank in provided dm range ####

T, D, W, S = modules.search_filterbank.search_fil(filename, lodm, hidm, max_width, threshold)

### Plot the results  ###

print(f"number of detection is {len(T)}")

### Convert DM to delay  #### 

header = modules.search_filterbank.headinfo(filename)
DM_delay = modules.search_filterbank.dm_to_delay(D, header.fmin, header.fmax)

if len(T) > 0:
  ### Convert DM to delay ####
  header = modules.search_filterbank.headinfo(filename)
  DM_delay = modules.search_filterbank.dm_to_delay(D, header.fmin, header.fmax)

  ### Do Clustering Dynamically Using Consistent Class Interface ####
  print(f"Running clustering using method: {cluster_method}")

  clusterer_map = {
      "fofw": lambda: modules.clustering_classes.fofW(T, DM_delay, S, W),
      "fof": lambda: modules.clustering_classes.fof(
          T, DM_delay, S, W, linking_length=args.fof_link
      ),
      "dbscan": lambda: modules.clustering_classes.DbscanClustering(
          T, DM_delay, S, W, eps=args.eps, min_samples=args.min_points
      ),
      "hdbscan": lambda: modules.clustering_classes.HdbscanClustering(
          T, DM_delay, S, W, min_cluster_size=args.min_points
      ),
  }

  clusterer = clusterer_map[cluster_method]()
  clusterer.do_clustering()
  clusters = clusterer.final_clusters()
  plotting = modules.plot_files.PlotAllClustersDMTime(T, D, W, S, clusters)
  plotting.plot_and_save("./", "DM_Time_clusters", show = True)

  print(f"Number of clusters found: {len(clusters)}")
else:
  print("No detections found to cluster.")
