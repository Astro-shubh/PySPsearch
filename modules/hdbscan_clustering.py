import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
import math

# Useful functions


def DM_to_Time(DM, nu1, nu2):
    return 4150000.0*(DM)*(1/(nu1*nu1) - 1/(nu2*nu2))

def Time_to_DM(Time, nu1, nu2):
    return Time/(4150000.0*(1/(nu1*nu1) - 1/(nu2*nu2)))

def MJD_to_Time(MJD):
    return (MJD-np.floor(MJD))*24.0*3600*1000.0

def Width_to_cluster(width, tsamp):
    return 8*0.064*np.log2(width/tsamp)

def cluster_to_width(width, tsamp):
    return tsamp*2.0**(width/tsamp)



######################     MAIN FUNCTION    ##################################
def get_clusters(filename, min_points, nu1, nu2):
    nu1 = nu1*1000.0    ### GHz to MHz
    nu2 = nu2*1000.0    ### GHz to MHz
    df = readfile(filename, nu1, nu2)
    return clusterer(df, min_points, nu1, nu2)



######################### Reading the SPCCL file    #############################
def readfile(filename, nu1, nu2):

    ##############   Read the file and store numbers in arrays  ###################
    f = open(str(filename), 'r')
    T = []
    D = []
    W = []
    S = []
    i = 0
    for line in f:
        if (i != 0):
            s = [float(r) for r in line.split()]
            T.append(s[0]*1000.0)    ### Changing second to ms
            D.append(s[1])
            W.append(s[2]*1000.0)    ###  Changing s to ms
            S.append(s[3])
        i = i+1

    T = np.array(T)
    D = np.array(D)
    D = DM_to_Time(D, nu1, nu2)


    ########################     Defining Dataframe   ############################

    df0 = pd.DataFrame({"Time": T, "DM": D, "Width": W, "SNR": S})
    df0 = df0.sort_values(by=["Time"])
    time = df0["Time"]
    return df0



####################   Function for Clustering   #######################################
def clusterer(df, mpts, nu1, nu2):

    #####################    Cluster the provided dataframe   #####################################
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mpts)
    clusterer.fit(df[["Time", "DM"]])
    df['labels'] = clusterer.labels_
    labels = df['labels'].to_numpy()
    uniq_id, indices = np.unique(labels, return_inverse=True)
    clusters = len(uniq_id)

    ####################   Storing the representative (brightest) detection for each cluster  #######################
    T_cl = []
    DM_cl = []
    W_cl = []
    id_cl = []
    S_cl = []
    for id in uniq_id:

        if (id != -1):
            locs = np.where(df['labels'].to_numpy() == id)[0]
            T = df["Time"].to_numpy()[locs]
            D = df["DM"].to_numpy()[locs]
            W = df["Width"].to_numpy()[locs]
            S = df["SNR"].to_numpy()[locs]
            x_max = np.where(S == max(S))[0][0]
            S_cl.append(S[x_max])
            T_cl.append(T[x_max])
            DM_cl.append(D[x_max])
            W_cl.append(W[x_max])
            id_cl.append(id)

    ####################     Width clustering, clusters that are within the detection-width distance, should have same label

    ##########################    Get the pairs of labels that should be same same  ###########################
    replace = []
    i = 0
    for i1 in id_cl:
        j = 0
        for i2 in id_cl:
            if (i1 != i2):
                if (np.sqrt((T_cl[i]-T_cl[j])**2.0+(DM_cl[i]-DM_cl[j])**2.0) < W_cl[i]):
                    replace.append([i1, i2])
                    id_cl.remove(i2)
            j = j+1
        i = i+1

    ########################     Replacing the labels    ###############################################
    T = df['Time'].to_numpy()
    W = df['Width'].to_numpy()
    S = df['SNR'].to_numpy()
    D = df['DM'].to_numpy()
    labels1 = labels
    for tup in replace:
        id1 = tup[1]
        id0 = tup[0]
        locs = np.where(labels1 == id1)[0]
        for j in locs:
            labels1[j] = id0

    #################        Making dataframe wth new labels     ########################################
    T = T/1000.0          ### ms to s
    W = W/1000.0          ###  ms to s
    dff = pd.DataFrame({"Time(s)": T, "DM": Time_to_DM(D, nu1, nu2), "Width(s)": W, "SNR": S})
    dff["Labels"] = labels1
    dff = dff.sort_values(by=["Labels"])
    return dff
