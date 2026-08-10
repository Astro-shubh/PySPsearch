import numpy as np
import scipy
import struct
import bottleneck as bn
import matplotlib.pyplot as plt
from multiprocessing import Pool

def prune_related1(hibins, hivals, downfact):              #### This function has been taken from the PRESTO's single_pulse_search.py
    # Remove candidates that are close to other candidates
    # but less significant.  This one works on the raw 
    # candidate arrays and uses the single downfact
    # that they were selected with.
    toremove = set()
    for ii in range(0, len(hibins)-1):
        if ii in toremove:  continue
        xbin, xsigma = hibins[ii], hivals[ii]
        for jj in range(ii+1, len(hibins)):
            ybin, ysigma = hibins[jj], hivals[jj]
            if (abs(ybin-xbin) > downfact//2):
                break
            else:
                if jj in toremove:
                    continue
                if (xsigma > ysigma):
                    toremove.add(jj)
                else:
                    toremove.add(ii)
    # Now zap them starting from the end
    toremove = sorted(toremove, reverse=True)
    for bin in toremove:
        del(hibins[bin])
        del(hivals[bin])
    return hibins, hivals





def detection(TimeSeries, BoxcarWidths, RmedWidth, Threshold, TimeSamp, TrialDM):
########   Declaring lists to store detections  ###########    

#    print("Doing detection on dm "+str(TrialDM))
    cand_Time = []
    cand_DM = []
    cand_Width = []
    cand_SNR = []
    TimeSeries = TimeSeries - bn.move.move_median(TimeSeries, window=RmedWidth, min_count=1)
    TimeSeries = TimeSeries - np.mean(TimeSeries)
    TimeSeries = TimeSeries/np.std(TimeSeries)
    for width in BoxcarWidths:
        if(width == 1):
            conv_result = TimeSeries
            centering = 0
        else:
#            kernel = np.ones(2*int(width/2.0)+1)
            conv_result = bn.move.move_sum(TimeSeries, window=width, min_count=1)
#            conv_result = np.convolve(TimeSeries, kernel, mode='same')
            centering = int(width/2.0)
        valid_start = width-1
        valid_end = len(conv_result)-width
        conv_result = conv_result[valid_start:]
        conv_result = conv_result/np.sqrt(float(width))
        detection_idx = np.where(conv_result > Threshold)[0]    ### Gives the index of detection in conv_result, which is equal to the index in Tseries - valid_start
        detection_snr = conv_result[detection_idx]             ###  Gives value of SNRs for the respective detections
        detection_idx, detection_snr = prune_related1(detection_idx.tolist(), detection_snr.tolist(), width)
        for i in range(len(detection_idx)):
            cand_Time.append((valid_start+detection_idx[i]+centering)*TimeSamp)  ### actual detection time is: start of valid conv (which is width-1) + 
                                                                                ####   detection idx in the dpricated conv + centering (width/2)
            cand_DM.append(TrialDM)
            cand_Width.append(width*TimeSamp)
            cand_SNR.append(detection_snr[i])

    return np.array(cand_Time), np.array(cand_DM), np.array(cand_Width), np.array(cand_SNR) 

