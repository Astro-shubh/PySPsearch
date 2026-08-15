from collections import deque
import hdbscan
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


from modules.fofw_wrap import PyfofW
import numpy as np


class fofW:
  """Friends-of-Friends clustering where linking length is determined by

  detection width, backed by a high-performance C++ Boost R-Tree engine.
  """

  def __init__(
      self,
      width_fraction: float = (
          0.7
      ),  # The fraction of actual width to use for linking length
      min_length: float = (
          0.01
      ),  # in seconds, Minimum linking length, avoids splitting of very narrow detections
      max_length: float = (
          0.5
      ),  # in seconds, Maximum linking length, avoids merging of wide detections
  ):
    self.width_fraction = float(width_fraction)
    self.min_length = float(min_length)
    self.max_length = float(max_length)
    self._final_clusters: list[list[int]] = []

  def do_clustering(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
  ):
    # Preprocess widths according to fraction, min_length, and max_length rules
    w_arr = np.array(widths, dtype=np.float32)
    processed_widths = self.width_fraction * w_arr
    processed_widths = np.clip(
        processed_widths, self.min_length, self.max_length
    )

    # Instantiate and run the fast C++ wrapper using the modified widths
    clusterer = PyfofW(
        x_locations=x_locations,
        y_locations=y_locations,
        snr=snr,
        widths=processed_widths,
    )
    clusterer.do_clustering()
    self._final_clusters = clusterer.final_clusters()

  def final_clusters(self) -> list[list[int]]:
    return self._final_clusters

class fofW_old:
  """Friends-of-Friends clustering where linking length is determined by detection width."""

  def __init__(
      self,
      width_fraction: float = 0.7,   ## The fraction of actual width to use for linking length
      min_length: float = 0.01,      ## in seconds, Minimum linking length, avoids slpitting of very narrow detections
      max_length: float = 0.5,       ## in seconds, Maximum linking length, avoids merging of wide detections
  ):
    self.width_fraction = float(width_fraction)
    self.min_length = float(min_length)
    self.max_length = float(max_length)
    self._final_clusters: list[list[int]] = []

  def do_clustering(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
  ):
    x = np.array(x_locations, dtype=float)
    y = np.array(y_locations, dtype=float)
    w_arr = np.array(widths, dtype=float)

    n = len(x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((x, y))
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

        current_w = self.width_fraction * w_arr[curr_idx]
        current_w = np.clip(current_w, self.min_length, self.max_length)
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
  """Conventional Friends-of-Friends (FoF) clustering using a constant linking length."""

  def __init__(self, linking_length: float):
    self.linking_length = float(linking_length)
    self._final_clusters: list[list[int]] = []

  def do_clustering(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
  ):
    x = np.array(x_locations, dtype=float)
    y = np.array(y_locations, dtype=float)

    n = len(x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((x, y))
    visited = np.zeros(n, dtype=bool)
    tree = KDTree(points)
    link_sq = self.linking_length * self.linking_length

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
        neighbors = tree.query_ball_point(curr_pt, r=self.linking_length)

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

  def __init__(self, eps: float, min_samples: int):
    self.eps = float(eps)
    self.min_samples = int(min_samples)
    self._final_clusters: list[list[int]] = []

  def do_clustering(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
  ):
    x = np.array(x_locations, dtype=float)
    y = np.array(y_locations, dtype=float)

    n = len(x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((x, y))

    model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
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
  """HDBSCAN clustering using the hdbscan library."""

  def __init__(self, min_cluster_size: int = 5, min_samples: int = None):
    self.min_cluster_size = int(min_cluster_size)
    self.min_samples = (
        int(min_samples) if min_samples is not None else None
    )
    self._final_clusters: list[list[int]] = []

  def do_clustering(
      self,
      x_locations: list[float],
      y_locations: list[float],
      snr: list[float],
      widths: list[float],
  ):
    x = np.array(x_locations, dtype=float)
    y = np.array(y_locations, dtype=float)

    n = len(x)
    if n == 0:
      self._final_clusters = []
      return

    self._final_clusters.clear()
    points = np.column_stack((x, y))

    model = hdbscan.HDBSCAN(
        min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
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
