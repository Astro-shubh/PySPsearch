import argparse
import numpy as np
#import multiprocessing as mp
import os
import sys
import glob
import pandas as pd
from concurrent.futures import wait
from concurrent.futures import ProcessPoolExecutor as Pool
import sigpyproc.readers
import sigpyproc.timeseries
import scipy.signal
import pandas as pd
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import modules.detection
import modules.hdbscan_clustering
from pyfdmt import transform

###########    Argument Parsing     #################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Input filterbank filename")
parser.add_argument("-lodm", "--low_dm", help="Smallest DM to search.")
parser.add_argument("-hidm", "--high_dm", help="Largest DM to search.")
parser.add_argument("-mw", "--max_width", help="Maximum width (s)")
parser.add_argument("-th", "--threshold", help="Detection threshold")
parser.add_argument("-mpt", "--min_points", help="min_points to use in the hdbscan clustering, clusters smaller than this would be recorded as noise")

args = parser.parse_args()
max_width = float(args.max_width)
filename = str(args.filename)
threshold = float(args.threshold)
min_points = int(args.min_points)
lodm = float(args.low_dm)
hidm = float(args.high_dm)

##########    Header information    #################
class headinfo:
	def __init__(self, filename):
		fil1 = sigpyproc.readers.FilReader(filename)
		self.nsamp = fil1.header.nsamples
		self.tsamp = fil1.header.tsamp        ## seconds
		self.fmax = fil1.header.fmax/1000.0    ## GHz
		self.fmin = fil1.header.fmin/1000.0    ## GHz
		self.nchan = fil1.header.nchans
		self.basename = fil1.header.basename

def generate_dmplan(header, dmlo, dmhi):
	dm_plan = []
	dm_const = 1/241.0
	nu1 = header.fmin
	nu2 = header.fmax
	del_t = header.tsamp
	chan_width = (nu2 - nu1)/header.nchan
	min_delay = 2*dm_const*lodm*chan_width/(nu1)**3.0
	print(f" Minimum delay {min_delay} for dm {lodm}")
	down_factor = np.ceil(min_delay/del_t)
	del_t = del_t*down_factor
	max_delay = 2*dm_const*hidm*chan_width/(nu1)**3.0
	if(max_delay <= del_t):
		dm_plan.append([lodm, hidm])
		return dm_plan
	else:
		dm1 = lodm
		while(del_t < max_delay):
			dm2 = (del_t)/(2*dm_const*chan_width/(nu1)**3.0)
			if(dm2 >= hidm):
				dm_plan.append([dm1, hidm])
				break;
			down_factor = down_factor*2
			del_t = del_t*2
			dm_plan.append([dm1, dm2])
			dm1 = dm2
		dm_plan.append([dm1, hidm])
		return dm_plan
	

########## lists to store the detections   ############

cand_Time = []
cand_DM = []
cand_Width = []
cand_SNR = []


header = headinfo(filename)
tsamp1 = header.tsamp
nsamp = header.nsamp
nchan = header.nchan
fmin = header.fmin
fmax = header.fmax
####   Filter widths   #########################

filters = []
max_filt_size = max_width/tsamp1
numfilts = int(np.log2(max_filt_size))+1
for i in range(numfilts+1):
    filters.append(2**i)
filters1 = np.array(filters)

#########   DM plan without dm step, the DM step is set by pyfdmt algortihm based on the frequency range and time resolution   ################


dm_plan = generate_dmplan(header, lodm, hidm)
#dm_plan = [[0.0,100],[100.0,300.0],[300.0,700.0],[700.0,1500.0],[1500.0,3100.0]]
#    dm_plan = [[700.0,1300.0]]
print(dm_plan)

#########    Filter block reading     ##################

overlap = (1/241.0)*3100.0*(1/(fmin*fmin) - 1.0/(fmax*fmax))  ## seconds

print(overlap)

overlap_samps = int(overlap/tsamp1)

buffer_size = 131072                    ### Fixed to Cheetah buffer size

iterations = int(np.floor(float(nsamp - overlap_samps)/float(buffer_size - overlap_samps))+1)



print("number of interations : "+str(iterations))
print(f"Low and frequency {fmin} {fmax}")
start = 0
fil = sigpyproc.readers.FilReader(filename)
#####################    Going through block iterations    ########################
for itr in range(iterations):
    filters = filters1
    if( (nsamp - start) < buffer_size):
        buffer_size = nsamp - start

    new_buffer = fil.read_block(start, buffer_size)
    start_time = start*tsamp1


    ##########    Renew the filter set for the current buffer      ###############
    
    filters = filters1

    #################   going through the DM plan     ###########################
    for plan_number in range(len(dm_plan)):
        print("working on DM range: "+str(dm_plan[plan_number])+'\n')
        down_factor = 2**plan_number
        using_buffer = new_buffer.downsample(1, down_factor)
        if(down_factor != 1):
            filters = np.delete(filters, len(filters)-1)

        ###########    Getting the block parameters    ##############
        tsamp = down_factor*tsamp1
        dm_time = transform(using_buffer.data, using_buffer.header.fmax, using_buffer.header.fmin, using_buffer.header.tsamp, dm_plan[plan_number][0], dm_plan[plan_number][1])
        DT_data = dm_time.data
        dm_list = dm_time.dms
        for dm_num in range(len(dm_list)):
            Tseries = DT_data[dm_num]
            rmed_width = 2.0*max_width/using_buffer.header.tsamp
            rmed_width = 2*int(rmed_width/2.0)+1
            T, D, W, S = modules.detection.detection(Tseries, filters, rmed_width, threshold, using_buffer.header.tsamp, dm_list[dm_num])
            for i in range(len(T)):
                cand_Time.append(T[i]+start_time)
                cand_DM.append(D[i])
                cand_Width.append(W[i])
                cand_SNR.append(S[i])
    #######  Setting the next start of the block_read  ##################
    start = start+buffer_size - overlap_samps


###########    Plotting the detections    ########################

detection_file = basename+'_detections.txt'
plt.title("Total number of detections: "+str(len(cand_Time)))
plt.xlabel("Time (s)")
plt.ylabel(" DM ")
plt.scatter(cand_Time, cand_DM)
plt.savefig(basename+"_detections_Time_DM.png")
plt.clf()

plt.title("Total number of detections: "+str(len(cand_Time)))
plt.xlabel("DM")
plt.ylabel(" SNR ")
plt.scatter(cand_DM, cand_SNR)
plt.savefig(basename+"_detections_DM_SNR.png")
plt.clf()


fd=open(detection_file, 'w')
fd.write("Time(s)  DM  Width(s)  SNR\n")
for i in range(len(cand_Time)):
    fd.write(str(cand_Time[i])+"  "+str(cand_DM[i])+"  "+str(cand_Width[i])+"  "+str(cand_SNR[i])+"\n")  


#############   Clustering the detections   ################

df1 = modules.hdbscan_clustering.get_clusters(detection_file, min_points, fmin, fmax)
cluster_file = basename+"_clusters.txt"
df1.to_csv(cluster_file, sep=" ")

############   Plotting the clusters    ####################
plt.title("Total number of clusters: "+str(len(np.unique(df1["Labels"].to_numpy()))))
plt.xlabel("Time (s)")
plt.ylabel("DM ")
plt.scatter(df1["Time(s)"].to_numpy(), df1["DM"].to_numpy(), c = df1["Labels"].to_numpy())
plt.savefig(basename+"_clusters.png")


