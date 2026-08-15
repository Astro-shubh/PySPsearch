from concurrent.futures import ProcessPoolExecutor, as_completed
import modules.detection
import numpy as np


def _run_block_clustering(
    clusterer_instance, t_chunk, dm_chunk, s_chunk, w_chunk, offset
):
  """Worker function to run clustering on a single block."""
  clusterer_instance.do_clustering(t_chunk, dm_chunk, s_chunk, w_chunk)
  block_clusters = clusterer_instance.final_clusters()

  global_clusters = []
  for clstr in block_clusters:
    global_clusters.append([idx + offset for idx in clstr])
  return global_clusters


def blockwise_clustering(
    clusterer_instance,
    time,
    dm,
    snr,
    width,
    chunk_size=100000,
    max_workers=4,
):
  """Splits large data into blocks and executes clustering in parallel."""
  time = np.asarray(time)
  dm = np.asarray(dm)
  snr = np.asarray(snr)
  width = np.asarray(width)

  n_total = len(time)
  if n_total <= chunk_size:
    slices = [slice(0, n_total)]
  else:
    print(f"Large dataset detected. Processing in parallel blocks of {chunk_size}...")
    slices = [
        slice(i, min(i + chunk_size, n_total))
        for i in range(0, n_total, chunk_size)
    ]

  clusters = []
  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for s in slices:
      futures.append(
          executor.submit(
              _run_block_clustering,
              clusterer_instance,
              time[s],
              dm[s],
              snr[s],
              width[s],
              s.start,
          )
      )

    for future in as_completed(futures):
      clusters.extend(future.result())

  return clusters
