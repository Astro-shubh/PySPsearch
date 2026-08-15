import numpy as np

def thresholding(T, D, W, S, clusters, Dm_threshold, mpt):
    Sifted_T = []
    Sifted_D = []
    Sifted_W = []
    Sifted_S = []
    for list in clusters:
        if(len(list) >= mpt):
            Time = T[list]
            Dm = D[list]
            Width = W[list]
            SNR = S[list]
            best_cand_idx = np.where(SNR == max(SNR))
            if(Dm[best_cand_idx] >= Dm_threshold):
                Sifted_T.append(Time[best_cand_idx])
                Sifted_D.append(Dm[best_cand_idx])
                Sifted_W.append(Width[best_cand_idx])
                Sifted_S.append(SNR[best_cand_idx])
        else:
            continue;
    Sifted_T = np.array(Sifted_T)
    Sifted_D = np.array(Sifted_D)
    Sifted_W = np.array(Sifted_W)
    Sifted_S = np.array(Sifted_S)
    return (Sifted_T, Sifted_D, Sifted_W, Sifted_S)
