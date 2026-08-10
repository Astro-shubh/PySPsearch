import matplotlib.pyplot as plt
import numpy as np
import os
from modules.write_products import read_FullSpccl, read_cluster


class PlotFullSpccl:
  """Plots the entire candidate space (Time, DM, Width, SNR) by reading a .spccl file directly."""

  def __init__(self, output_folder, output_filebase):
    self.output_folder = output_folder
    self.output_filebase = output_filebase
    
    # Load data automatically using the reader function
    self.time, self.dm, self.width, self.snr = read_FullSpccl(
        output_folder, output_filebase
    )

  def plot_and_save(self, show=False):
    full_filename = os.path.join(
        self.output_folder, f"{self.output_filebase}_full_diagnostic.png"
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # 1. Time vs DM (Color = SNR)
    sc1 = axes[0].scatter(
        self.time, self.dm, c=self.snr, cmap="viridis", s=10, alpha=0.8
    )
    axes[0].set_ylabel("Dispersion Measure (pc cm$^{-3}$)")
    axes[0].set_title("Time vs DM (Colored by S/N)")
    cbar1 = fig.colorbar(sc1, ax=axes[0])
    cbar1.set_label("S/N")

    # 2. DM vs SNR (Color = Width)
    sc2 = axes[1].scatter(
        self.dm, self.snr, c=self.width, cmap="plasma", s=10, alpha=0.8
    )
    axes[1].set_xlabel("Dispersion Measure (pc cm$^{-3}$)")
    axes[1].set_ylabel("S/N")
    axes[1].set_title("DM vs S/N (Colored by Width)")
    cbar2 = fig.colorbar(sc2, ax=axes[1])
    cbar2.set_label("Width (s)")

    # 3. Time vs SNR (Color = Width)
    sc3 = axes[2].scatter(
        self.time, self.snr, c=self.width, cmap="plasma", s=10, alpha=0.8
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("S/N")
    axes[2].set_title("Time vs S/N (Colored by Width)")
    cbar3 = fig.colorbar(sc3, ax=axes[2])
    cbar3.set_label("Width (s)")

    plt.tight_layout()
    plt.savefig(full_filename, dpi=300)
    print(f"Saved full diagnostic plot to {full_filename}")

    if show:
      plt.show()
    else:
      plt.close(fig)


class PlotCluster:
  """Generates a multi-panel diagnostic plot by reading an individual .cluster file directly."""

  def __init__(self, cluster_filepath):
    self.cluster_filepath = cluster_filepath
    
    # Load data automatically using the reader function
    self.time, self.dm, self.width, self.snr = read_cluster(cluster_filepath)

  def plot_and_save(self, show=False):
    # Derive output paths from the input cluster filename
    folder, filename = os.path.split(self.cluster_filepath)
    base_name = os.path.splitext(filename)[0]
    full_filename = os.path.join(folder, f"{base_name}_cluster_panel.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Time vs DM (Colored by S/N)
    sc1 = axes[0, 0].scatter(
        self.time, self.dm, c=self.snr, cmap="viridis", s=25, alpha=0.9
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("DM (pc cm$^{-3}$)")
    axes[0, 0].set_title("Time vs DM")
    fig.colorbar(sc1, ax=axes[0, 0], label="S/N")

    # Panel 2: DM vs S/N (Colored by Width)
    sc2 = axes[0, 1].scatter(
        self.dm, self.snr, c=self.width, cmap="plasma", s=25, alpha=0.9
    )
    axes[0, 1].set_xlabel("DM (pc cm$^{-3}$)")
    axes[0, 1].set_ylabel("S/N")
    axes[0, 1].set_title("DM vs S/N")
    fig.colorbar(sc2, ax=axes[0, 1], label="Width (s)")

    # Panel 3: Time vs S/N (Colored by Width)
    sc3 = axes[1, 0].scatter(
        self.time, self.snr, c=self.width, cmap="plasma", s=25, alpha=0.9
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("S/N")
    axes[1, 0].set_title("Time vs S/N")
    fig.colorbar(sc3, ax=axes[1, 0], label="Width (s)")

    # Panel 4: Time vs Width (Colored by S/N)
    sc4 = axes[1, 1].scatter(
        self.time, self.width, c=self.snr, cmap="viridis", s=25, alpha=0.9
    )
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Width (s)")
    axes[1, 1].set_title("Time vs Width")
    fig.colorbar(sc4, ax=axes[1, 1], label="S/N")

    plt.tight_layout()
    plt.savefig(full_filename, dpi=300)
    print(f"Saved cluster multi-panel plot to {full_filename}")

    if show:
      plt.show()
    else:
      plt.close(fig)

class PlotAllClustersDMTime:
  """Plots the entire DM vs Time plane, highlighting different clusters in distinct colors

  and showing unclustered points as background noise.
  """

  def __init__(self, time, dm, width, snr, clusters):
    self.time = np.asarray(time)
    self.dm = np.asarray(dm)
    self.width = np.asarray(width)
    self.snr = np.asarray(snr)
    self.clusters = clusters  # List of index lists [ [cluster_0_idxs], [cluster_1_idxs], ... ]

  def plot_and_save(self, output_folder, output_filebase, show=False):
    os.makedirs(output_folder, exist_ok=True)
    full_filename = os.path.join(
        output_folder, f"{output_filebase}_all_clusters_DM_Time.png"
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. Identify all points that belong to any cluster
    clustered_indices = set()
    for cluster_idxs in self.clusters:
      clustered_indices.update(cluster_idxs)

    # 2. Identify noise/unclustered points (everything else)
    all_indices = set(range(len(self.time)))
    noise_indices = list(all_indices - clustered_indices)

    # Plot background noise points first (gray, small, transparent)
    if noise_indices:
      ax.scatter(
          self.time[noise_indices],
          self.dm[noise_indices],
          c="lightgray",
          s=10,
          alpha=0.4,
          label="Noise / RFI",
          zorder=1,
      )

    # 3. Plot each cluster with a unique color from a matplotlib colormap
    # Using 'tab20' or 'gist_ncar' to give distinct colors for multiple clusters
    cmap = plt.get_cmap("tab20")
    num_clusters = len(self.clusters)

    for i, cluster_idxs in enumerate(self.clusters):
      if len(cluster_idxs) == 0:
        continue

      # Cycle through colormap colors
      color = cmap(i % 20)

      ax.scatter(
          self.time[cluster_idxs],
          self.dm[cluster_idxs],
          color=color,
          s=25,
          alpha=0.9,
          edgecolors="none",
          label=f"Cluster {i} (n={len(cluster_idxs)})",
          zorder=2,
      )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Dispersion Measure (pc cm$^{-3}$)", fontsize=12)
    ax.set_title(
        f"DM vs Time - Clustered Candidates ({num_clusters} clusters found)",
        fontsize=14,
    )

    # Add a clean legend if there aren't an overwhelming number of clusters
    if num_clusters <= 15:
      ax.legend(
          loc="upper right",
          bbox_to_anchor=(1.15, 1.0),
          fontsize=9,
          frameon=True,
      )
    else:
      # If there are too many clusters, skip the massive legend to keep the plot readable
      ax.text(
          0.02,
          0.95,
          f"Total Clusters: {num_clusters}",
          transform=ax.transAxes,
          fontsize=10,
          verticalalignment="top",
          bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
      )

    plt.tight_layout()
    plt.savefig(full_filename, dpi=300)
    print(f"Saved global cluster DM-Time plot to {full_filename}")

    if show:
      plt.show()
    else:
      plt.close(fig)
