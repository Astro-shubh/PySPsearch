from concurrent.futures import ProcessPoolExecutor
import bottleneck as bn
import modules.detection
import modules.parallel_clustering
import modules.sifting_classes
import numpy as np
import pandas as pd
from pyfdmt import transform
import scipy.signal
import sigpyproc.readers
import sigpyproc.timeseries
import time


def dm_to_delay(dm, fmin, fmax):
  dm = np.asarray(dm, dtype=float)
  return (1 / 241.0) * dm * (1 / (fmin * fmin) - 1.0 / (fmax * fmax))


class headinfo:

  def __init__(self, filename):
    fil1 = sigpyproc.readers.FilReader(filename)
    self.nsamp = fil1.header.nsamples
    self.tsamp = fil1.header.tsamp
    self.fmax = fil1.header.fmax / 1000.0
    self.fmin = fil1.header.fmin / 1000.0
    self.nchan = fil1.header.nchans
    self.basename = fil1.header.basename


def generate_dmplan(header, lodm, hidm):
  if(lodm <= 0.1):
    lodm = 0.1
  dm_plan = []
  down_plan = []
  dm_const = 1 / 241.0
  nu1 = header.fmin
  nu2 = header.fmax
  del_t = header.tsamp
  chan_width = (nu2 - nu1) / header.nchan
  min_delay = 2 * dm_const * lodm * chan_width / (nu1) ** 3.0
  down_factor = np.ceil(min_delay / del_t)
  del_t = del_t * down_factor
  max_delay = 2 * dm_const * hidm * chan_width / (nu1) ** 3.0
  if max_delay <= del_t:
    dm_plan.append([lodm, hidm])
    down_plan.append(int(down_factor))
    return dm_plan, down_plan
  else:
    dm1 = lodm
    while del_t < max_delay:
      dm2 = del_t / (2 * dm_const * chan_width / (nu1) ** 3.0)
      if dm2 >= hidm:
        dm_plan.append([dm1, hidm])
        down_plan.append(int(down_factor))
        down_factor = down_factor * 2
        break
      dm_plan.append([dm1, dm2])
      down_plan.append(int(down_factor))
      down_factor = down_factor * 2
      del_t = del_t * 2
      dm1 = dm2
    dm_plan.append([dm1, hidm])
    down_plan.append(int(down_factor))
    return dm_plan, down_plan


def get_filters(max_width, tsamp):
  filters = []
  max_filt_size = max_width / tsamp
  num_filters = int(np.log2(max_filt_size)) + 1
  for i in range(num_filters + 1):
    filters.append(2 ** i)
  return filters


# Worker wrapper function to process a single DM time series independently
def process_single_dm(args):
  Tseries, filters, rmed_width, threshold, tsamp, dm_val, rms = args
  # Call detection serially for this specific DM (using the fast bottleneck code)
  return modules.detection.detection(
      Tseries, filters, rmed_width, threshold, tsamp, dm_val, rms
  )


def search_fil(filename, lodm, hidm, max_width, threshold, configured_clusterer, out_dir, write_clusters):
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

  print("Generating DM plan...")
  dm_plan, down_plan = generate_dmplan(header, lodm, hidm)
  print("DM ranges from DM plan: ", dm_plan)
  print("Downsampling for each DM range: ", down_plan)

  overlap = dm_to_delay(hidm, fmin, fmax)
  overlap_samps = int(overlap / tsamp1)
  chunk_to_process = 5.0
  chunk_samples = int(chunk_to_process / tsamp1)
  buffer_size = chunk_samples + overlap_samps

  iterations = int(np.floor(float(nsamp - overlap_samps) / float(chunk_samples)) + 1)
  print("number of iterations : " + str(iterations))

  start = 0     # Start sample for reading the filterbank
  fil = sigpyproc.readers.FilReader(filename)

  # Create a persistent process pool executor restricted to max 8 cores
  with ProcessPoolExecutor(max_workers=8) as executor:
    for itr in range(iterations):

      # List to store the detections in current interattion
      itr_Time = []
      itr_DM = []
      itr_SNR = []
      itr_Width = []

      #  If available number of samples is smaller than the buffer size
      if (nsamp - start) < buffer_size:
        buffer_size = nsamp - start

      # read the chunk of buffer size
      new_buffer = fil.read_block(start, buffer_size)
      start_time = start * tsamp1
      print(f"Processing buffer number: {itr}")
      detection_start = time.perf_counter()

      # Go through each DM plan
      for plan_number in range(len(dm_plan)):
        print("working on DM range: " + str(dm_plan[plan_number]) + "\n")
        # Get the downsampling for current DM range
        down_factor = down_plan[plan_number]
        # Downsample the buffer to current downfactor
        using_buffer = new_buffer.downsample(1, down_factor)
        # Compute filters for current downsampling
        filters = get_filters(max_width, using_buffer.header.tsamp)

        # Compute the dedispersion trnasform
        tsamp = using_buffer.header.tsamp
        dm_time = transform(
            using_buffer.data,
            using_buffer.header.fmax,
            using_buffer.header.fmin,
            tsamp,
            dm_plan[plan_number][0],
            dm_plan[plan_number][1],
        )
        print(
            f"Done with dedispersion. Total number of DMs: {len(dm_time.dms)}"
        )

        # Get the dedispersed data and dm list
        DT_data = dm_time.data
        dm_list = dm_time.dms
        is_last_iteration = itr == (iterations - 1)

        # Decide the valid time range in DT data (i.e. exclude the overlap part
        if is_last_iteration:
            valid_bins = DT_data.shape[1]
        else:
            valid_bins = int(chunk_samples / down_factor)
        DT_data = DT_data[:, :valid_bins]

        # Cmpute the rms from lowest DM time series
        rms = np.std(DT_data[0])

        # Compute the running median width based on max search width
        rmed_width = 2.0 * max_width / tsamp
        rmed_width = 2 * int(rmed_width / 2.0) + 1

        # Prepare tasks for all DMs in this block
        tasks = [
            (DT_data[dm_num], filters, rmed_width, threshold, tsamp, dm_list[dm_num], rms)
            for dm_num in range(len(dm_list))
        ]

        # Execute detection across the 8-core process pool for all DMs simultaneously
        results = list(executor.map(process_single_dm, tasks))

        # write the candidates of the current DM range to the iteration results    
        for T, D, W, S in results:
          for ii in range(len(T)):
            itr_Time.append(T[ii])
            itr_DM.append(D[ii])
            itr_Width.append(W[ii])
            itr_SNR.append(S[ii])

      detection_time = time.perf_counter() - detection_start
      print(f"-> detection completed in {detection_time:.2f} seconds.")

      #  Cluster the detections of current iteration
      DM_delay = modules.search_filterbank.dm_to_delay(itr_DM, header.fmin, header.fmax)
      print(f"Started clustering on {len(itr_Time)} detection in iteration {itr}")
      clustering_start = time.perf_counter()
      clusters = modules.parallel_clustering.blockwise_clustering(configured_clusterer, itr_Time, DM_delay, itr_SNR, itr_Width, chunk_size=1000000)
      clustering_time = time.perf_counter() - clustering_start
      print(f"-> clustering completed in {clustering_time:.2f} seconds.")

      #write the clusters if asked
      if(write_clusters):
        modules.write_products.write_clusters(itr_Time, itr_DM, itr_Width, itr_SNR, clusters, out_dir, header.basename)

      # Sift the clusters of the current itr
      sifted_T, sifted_D, sifted_W, sifted_S = modules.sifting_classes.thresholding(itr_Time, itr_DM, itr_Width, itr_SNR, clusters, 1.0, 10)

      # Store the sifted results of current itr to global candidate list
      for jj in range(len(sifted_T)):
        cand_Time.append(sifted_T[jj])
        cand_DM.append(sifted_D[jj])
        cand_Width.append(sifted_W[jj])
        cand_SNR.append(sifted_S[jj])

      # set the new start for reading the filterbank file for next iteration
      start = start + buffer_size - overlap_samps
      print(
          f"Found {len(itr_Time)} detections and {len(sifted_T)} candidate in buffer"
          f" {itr}"
      )
#      if(itr >= 2):
#          break;

  return cand_Time, cand_DM, cand_Width, cand_SNR
