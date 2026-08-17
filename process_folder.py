import argparse
import os
import subprocess
import sys
import tracemalloc

parser = argparse.ArgumentParser(
    description="Perform single pulse search on all filterbank files in a directory"
)

parser.add_argument(
    "-d", "--fil-directory", type=str, help="Directory with filterbanks"
)
parser.add_argument(
    "-lodm", "--low_dm", type=float, help="Smallest DM to search."
)
parser.add_argument(
    "-hidm", "--high_dm", type=float, help="Largest DM to search."
)
parser.add_argument("-mw", "--max_width", type=float, help="Maximum width (s)")
parser.add_argument("-th", "--threshold", type=float, help="Detection threshold")
parser.add_argument(
    "-od",
    "--output-directory",
    type=str,
    default="./",
    help="Directory path to save cluster files",
)

args = parser.parse_args()

in_dir = args.fil_directory
out_dir = args.output_directory
lodm = args.low_dm
hidm = args.high_dm
mw = args.max_width
th = args.threshold

software_dir = (
    "/raid/Shubham_files/RFSIFT_investigation/spccl_from_PySPsearch/PySPsearch/"
)
pipeline_script = os.path.join(software_dir, "search_pipeline.py")

extension = ".fil"

files = [
    os.path.join(in_dir, f)
    for f in os.listdir(in_dir)
    if f.endswith(extension) and os.path.isfile(os.path.join(in_dir, f))
]

for file in files:
  print(f"Working on file {file}\n")
  print("starting PySPsearch run\n")

  # Build command as a list for subprocess
  cmd = [
      sys.executable,
      pipeline_script,
      "-f",
      file,
      "-lodm",
      str(lodm),
      "-hidm",
      str(hidm),
      "-mw",
      str(mw),
      "-th",
      str(th),
      "-mpt",
      "10",
      "-cm",
      "fofW",
      "--output-directory",
      out_dir,
      "--write-clusters",
  ]
  tracemalloc.start()
  # Execute and wait for each file to finish completely before starting the next one
  try:
    subprocess.run(cmd, check=True)
  except subprocess.CalledProcessError as e:
    print(f"Error processing file {file}: {e}")
    # Optional: decide whether to continue or break if one file fails
  snapshot = tracemalloc.take_snapshot()
  top_stats = snapshot.statistics("lineno")
  print("[ Top 10 Memory Allocating Lines ]")
  for stat in top_stats[:10]:
    print(stat)
print(f"Processed all filterbanks in directory: {in_dir}")
