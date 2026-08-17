import numpy as np


def thresholding(T, D, W, S, clusters, Dm_threshold, mpt):
  # Ensure input arrays are numpy arrays so advanced indexing works reliably
  T = np.asarray(T)
  D = np.asarray(D)
  W = np.asarray(W)
  S = np.asarray(S)

  Sifted_T = []
  Sifted_D = []
  Sifted_W = []
  Sifted_S = []

  for cluster_indices in clusters:
    # Convert cluster indices to a numpy array of integers just in case
    cluster_indices = np.array(cluster_indices, dtype=int)

    if len(cluster_indices) >= mpt:
      # Extract values for this cluster
      Time = T[cluster_indices]
      Dm = D[cluster_indices]
      Width = W[cluster_indices]
      SNR = S[cluster_indices]

      # Find the index of the best candidate (highest SNR) in this cluster
      best_idx = np.argmax(SNR)

      # Check DM threshold against the best candidate
      if Dm[best_idx] >= Dm_threshold:
        Sifted_T.append(Time[best_idx])
        Sifted_D.append(Dm[best_idx])
        Sifted_W.append(Width[best_idx])
        Sifted_S.append(SNR[best_idx])

  return (
      np.array(Sifted_T),
      np.array(Sifted_D),
      np.array(Sifted_W),
      np.array(Sifted_S),
  )
