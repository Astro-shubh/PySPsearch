from collections import deque
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import hdbscan


class fofW:
  """Friends-of-Friends clustering where linking length is determined by point width."""

  def __init__(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
  ):
    self._widths = np.array(widths, dtype=float)
    self._x = np.array(x_locations, dtype=float)
    self._y = np.array(y_locations, dtype=float)
    self._snr = np.array(snr, dtype=float)
    self._final_clusters: list[list[int]] = []

  def do_clustering(self):
    n = len(self._x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((self._x, self._y))
    visited = np.zeros(n, dtype=bool)
    tree = KDTree(points)

    for i in range(n):
      if visited[i]:
        continue

      current_final_cluster = []
      traversal_queue = deque([i])
      visited[i] = True

      while traversal_queue:
        curr_idx = traversal_queue.popleft()
        current_final_cluster.append(curr_idx)

        current_w = self._widths[curr_idx]
        curr_pt = points[curr_idx]

        neighbors = tree.query_ball_point(curr_pt, r=current_w, p=float("inf"))

        for neighbor_idx in neighbors:
          if not visited[neighbor_idx]:
            dx = curr_pt[0] - points[neighbor_idx, 0]
            dy = curr_pt[1] - points[neighbor_idx, 1]
            if (dx * dx + dy * dy) < (current_w * current_w):
              visited[neighbor_idx] = True
              traversal_queue.append(neighbor_idx)

      self._final_clusters.append(current_final_cluster)

  def final_clusters(self) -> list[list[int]]:
    return self._final_clusters


class fof:
  """Conventional Friends-of-Friends (FoF) clustering using a constant linking length

  accelerated by a KD-Tree.
  """

  def __init__(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
      linking_length: float,
  ):
    self._widths = np.array(widths, dtype=float)
    self._x = np.array(x_locations, dtype=float)
    self._y = np.array(y_locations, dtype=float)
    self._snr = np.array(snr, dtype=float)
    self._linking_length = float(linking_length)
    self._final_clusters: list[list[int]] = []

  def do_clustering(self):
    n = len(self._x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((self._x, self._y))
    visited = np.zeros(n, dtype=bool)
    tree = KDTree(points)
    link_sq = self._linking_length * self._linking_length

    for i in range(n):
      if visited[i]:
        continue

      current_final_cluster = []
      traversal_queue = deque([i])
      visited[i] = True

      while traversal_queue:
        curr_idx = traversal_queue.popleft()
        current_final_cluster.append(curr_idx)

        curr_pt = points[curr_idx]
        neighbors = tree.query_ball_point(curr_pt, r=self._linking_length)

        for neighbor_idx in neighbors:
          if not visited[neighbor_idx]:
            dx = curr_pt[0] - points[neighbor_idx, 0]
            dy = curr_pt[1] - points[neighbor_idx, 1]
            if (dx * dx + dy * dy) <= link_sq:
              visited[neighbor_idx] = True
              traversal_queue.append(neighbor_idx)

      self._final_clusters.append(current_final_cluster)

  def final_clusters(self) -> list[list[int]]:
    return self._final_clusters


class DbscanClustering:
  """DBSCAN clustering using user-provided epsilon and minimum samples."""

  def __init__(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
      eps: float,
      min_samples: int,
  ):
    self._widths = np.array(widths, dtype=float)
    self._x = np.array(x_locations, dtype=float)
    self._y = np.array(y_locations, dtype=float)
    self._snr = np.array(snr, dtype=float)
    self._eps = float(eps)
    self._min_samples = int(min_samples)
    self._final_clusters: list[list[int]] = []

  def do_clustering(self):
    n = len(self._x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((self._x, self._y))

    model = DBSCAN(eps=self._eps, min_samples=self._min_samples)
    labels = model.fit_predict(points)

    clusters_map = {}
    for idx, label in enumerate(labels):
      if label == -1:
        continue
      if label not in clusters_map:
        clusters_map[label] = []
      clusters_map[label].append(idx)

    self._final_clusters = list(clusters_map.values())

  def final_clusters(self) -> list[list[int]]:
    return self._final_clusters


class HdbscanClustering:
  """HDBSCAN clustering using the installed hdbscan library."""

  def __init__(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
      min_cluster_size: int = 5,
      min_samples: int = None,
  ):
    self._widths = np.array(widths, dtype=float)
    self._x = np.array(x_locations, dtype=float)
    self._y = np.array(y_locations, dtype=float)
    self._snr = np.array(snr, dtype=float)
    self._min_cluster_size = int(min_cluster_size)
    self._min_samples = (
        int(min_samples) if min_samples is not None else None
    )
    self._final_clusters: list[list[int]] = []

  def do_clustering(self):
    n = len(self._x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((self._x, self._y))

    model = hdbscan.HDBSCAN(
        min_cluster_size=self._min_cluster_size, min_samples=self._min_samples
    )
    labels = model.fit_predict(points)

    clusters_map = {}
    for idx, label in enumerate(labels):
      if label == -1:
        continue
      if label not in clusters_map:
        clusters_map[label] = []
      clusters_map[label].append(idx)

    self._final_clusters = list(clusters_map.values())

  def final_clusters(self) -> list[list[int]]:
    return self._final_clusters
