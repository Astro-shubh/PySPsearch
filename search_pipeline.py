import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import matplotlib.pyplot as plt
import modules.clustering_classes
import modules.plot_files
import modules.search_filterbank
import modules.parallel_clustering
import modules.write_products
import numpy as np
import os
import pandas as pd
import sys
import time

###########    Argument Parsing     #################

parser = argparse.ArgumentParser(
    description="Single-pulse search and parallel block-wise clustering pipeline"
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

parser.add_argument(
    "--write-clusters",
    action="store_true",
    help="Enable writing individual clusters to disk",
)
parser.add_argument(
    "--output-directory",
    type=str,
    default="./",
    help="Directory path to save cluster files",
)

args = parser.parse_args()
max_width = float(args.max_width)
filename = str(args.filename)
threshold = float(args.threshold)
lodm = float(args.low_dm)
hidm = float(args.high_dm)
cluster_method = args.cluster_method.lower()
out_dir = args.output_directory

### Search the filterbank in provided dm range ####

print("Starting search...")
search_start = time.perf_counter()

print(
    f"Configuring clustering instance for method: {cluster_method}"
)

if cluster_method == "fofw":
  configured_clusterer = modules.clustering_classes.fofW(
      width_fraction=1.0, min_length=0.01, max_length=0.5
   )
elif cluster_method == "fof":
  configured_clusterer = modules.clustering_classes.fof(
      linking_length=args.fof_link
  )
elif cluster_method == "dbscan":
  configured_clusterer = modules.clustering_classes.DbscanClustering(
      eps=args.eps, min_samples=args.min_points
  )
elif cluster_method == "hdbscan":
  configured_clusterer = modules.clustering_classes.HdbscanClustering(
      min_cluster_size=args.min_points
  )

T, D, W, S = modules.search_filterbank.search_fil(filename, lodm, hidm, max_width, threshold, configured_clusterer, out_dir, args.write_clusters)

search_duration = time.perf_counter() - search_start
print(f"-> Search completed in {search_duration:.2f} seconds.")

print(f"number of pulses found is {len(T)}")
