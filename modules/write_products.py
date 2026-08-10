import os
import numpy as np
import pandas as pd


def write_FullSpccl(Time, DM, Width, SNR, output_folder, output_filebase):
  """Writes all candidate detections to a full .spccl file."""
  os.makedirs(output_folder, exist_ok=True)
  full_filename = os.path.join(output_folder, f"{output_filebase}.spccl")

  df = pd.DataFrame(
      {"Time": Time, "DM": DM, "Width": Width, "SNR": SNR}
  )
  df.to_csv(full_filename, sep=" ", index=False, float_format="%.6f")
  print(f"Wrote detections to {full_filename}")


def read_FullSpccl(output_folder, output_filebase):
  """Reads a full .spccl candidate file and returns Time, DM, Width, SNR as arrays."""
  full_filename = os.path.join(output_folder, f"{output_filebase}.spccl")

  if not os.path.exists(full_filename):
    raise FileNotFoundError(f"Could not find candidate file: {full_filename}")

  df = pd.read_csv(full_filename, delim_whitespace=True)

  return (
      df["Time"].values,
      df["DM"].values,
      df["Width"].values,
      df["SNR"].values,
  )


def write_clusters(
    Time, DM, Width, SNR, clusters, output_folder, output_filebase
):
  """Writes individual cluster groupings out to separate .cluster files."""
  os.makedirs(output_folder, exist_ok=True)

  for i, cluster_indices in enumerate(clusters):
    full_filename = os.path.join(
        output_folder, f"{output_filebase}_{i}.cluster"
    )

    df_cluster = pd.DataFrame({
        "Time": np.array(Time)[cluster_indices],
        "DM": np.array(DM)[cluster_indices],
        "Width": np.array(Width)[cluster_indices],
        "SNR": np.array(SNR)[cluster_indices],
    })

    df_cluster.to_csv(full_filename, sep=" ", index=False, float_format="%.6f")

  print(f"Wrote {len(clusters)} cluster files to {output_folder}")


def read_cluster(cluster_filepath):
  """Reads a single .cluster file and returns Time, DM, Width, SNR as arrays."""
  if not os.path.exists(cluster_filepath):
    raise FileNotFoundError(f"Could not find cluster file: {cluster_filepath}")

  df = pd.read_csv(cluster_filepath, delim_whitespace=True)

  return (
      df["Time"].values,
      df["DM"].values,
      df["Width"].values,
      df["SNR"].values,
  )
