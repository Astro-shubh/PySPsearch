import numpy as np
import scipy
import struct
import bottleneck as bn
from numba import njit


@njit(fastmath=True)
def prune_related_numba(hibins, hivals, downfact):
  """Numba-accelerated version of PRESTO's candidate pruning logic.
  """
  n = len(hibins)
  if n <= 1:
    return hibins, hivals

  keep = np.ones(n, dtype=np.bool_)
  radius = downfact // 2

  for ii in range(n):
    if not keep[ii]:
      continue

    xbin = hibins[ii]
    xsigma = hivals[ii]

    # Look forward at subsequent candidates
    for jj in range(ii + 1, n):
      ybin = hibins[jj]

      # Since hibins is sorted, if distance exceeds radius, we can break early
      if (ybin - xbin) > radius:
        break

      if not keep[jj]:
        continue

      ysigma = hivals[jj]

      # Keep the one with higher sigma (S/N)
      if xsigma >= ysigma:
        keep[jj] = False
      else:
        keep[ii] = False
        break  # ii is dropped, no need to check further for ii

  # Filter arrays using boolean mask (very fast in NumPy/Numba)
  return hibins[keep], hivals[keep]


@njit(fastmath=True)
def _process_boxcar_detections(
    conv_result, width, threshold, valid_start, centering, time_samp, trial_dm
):
  """Function to perform detection on the normalized convolution results."""
  n = len(conv_result)
  # Count how many pass threshold to pre-allocate exact sizes
  count = 0
  for i in range(n):
    if conv_result[i] > threshold:
      count += 1

  if count == 0:
    return (
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )

  det_indices = np.empty(count, dtype=np.int64)
  det_snrs = np.empty(count, dtype=np.float64)
  # get all the indices and SNRs above threhsold
  idx = 0
  for i in range(n):
    if conv_result[i] > threshold:
      det_indices[idx] = i
      det_snrs[idx] = conv_result[i]
      idx += 1

  # Prune related inline or call numba pruning
  det_indices, det_snrs = prune_related_numba(det_indices, det_snrs, width)
  
  # Create vectors to hold the time, dm, width axis of pruned detections (prune dS/N is already available
  m = len(det_indices)
  times = np.empty(m, dtype=np.float64)
  # dm and width is same for all these detections
  dms = np.full(m, trial_dm, dtype=np.float64)
  widths_out = np.full(m, width * time_samp, dtype=np.float64)
  # fill the time axis using the pruned indices
  for i in range(m):
    times[i] = (valid_start + det_indices[i] + centering) * time_samp

  return times, dms, widths_out, det_snrs


def detection(
    TimeSeries, BoxcarWidths, RmedWidth, Threshold, TimeSamp, TrialDM, rms
):
  # Baseline subtraction, using moving median from bottleneck library
  TimeSeries = TimeSeries - bn.move.move_median(
      TimeSeries, window=RmedWidth, min_count=1
  )
  # Nromalizing to unit rms
  TimeSeries = TimeSeries / rms
  # List to store pruned arrays for individual trial widths
  all_times = []
  all_dms = []
  all_widths = []
  all_snrs = []
  # Searchign over trial widths
  for width in BoxcarWidths:
    if width == 1:
      conv_result = TimeSeries
      centering = 0
    else:
      # Running summation for current width
      conv_result = bn.move.move_sum(TimeSeries, window=width, min_count=1)
      # Summation result for a lag will be sotred at the begining of the box, but it represents an event in the middle of the box
      centering = int(width / 2.0)
    # The summation for width w is onyl valid after a lag of w (only then the box is filled
    valid_start = width - 1
    # get the valid convlution resutl only 
    conv_result = conv_result[valid_start:-width]
    # Since it was summation and initial time series was unit RMS current RMS is RMS*sqrt(width), S/N = sum/sqrt(width)
    conv_result = conv_result / np.sqrt(float(width))

    # Run the detection and pruning part
    t, d, w, s = _process_boxcar_detections(
        conv_result,
        width,
        Threshold,
        valid_start,
        centering,
        TimeSamp,
        TrialDM,
    )
    # If detections are returned, store the vectors in the predecided lists
    if len(t) > 0:
      all_times.append(t)
      all_dms.append(d)
      all_widths.append(w)
      all_snrs.append(s)

  # Concatenate all boxcar width results at once
  if len(all_times) > 0:
    return (
        np.concatenate(all_times),
        np.concatenate(all_dms),
        np.concatenate(all_widths),
        np.concatenate(all_snrs),
    )
  else:
    return (
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
    )
