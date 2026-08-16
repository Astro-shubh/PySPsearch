import os
import sys
import argparse

parser = argparse.ArgumentParser(
    description="Perform single pulse search on all filterbank files in a directory"
)

parser.add_argument("-d", "--fil-directory", type=str, help="Directory with filterbanks")
parser.add_argument("-lodm", "--low_dm", type=float,  help="Smallest DM to search.")
parser.add_argument("-hidm", "--high_dm", type=float,  help="Largest DM to search.")
parser.add_argument("-mw", "--max_width", type=float,  help="Maximum width (s)")
parser.add_argument("-th", "--threshold", type=float,  help="Detection threshold")
parser.add_argument("-od", "--output-directory", type=str, default="./", help="Directory path to save cluster files")

args = parser.parse_args()

in_dir = args.fil_directory
out_dir = args.output_directory
lodm = args.low_dm
hidm = args.high_dm
mw = args.max_width
th = args.threshold

software_dir = "/raid/Shubham_files/RFSIFT_investigation/spccl_from_PySPsearch/PySPsearch/"

extension = '.fil'

files = [
    os.path.join(in_dir, f)
    for f in os.listdir(in_dir)
    if f.endswith(extension) and os.path.isfile(os.path.join(in_dir, f))
]

for file in files:
    print('Working on file '+file+'\n')
    print('starting PySPsearch run\n')
    os.system(f" python {software_dir}search_pipeline.py -f {file} -lodm {lodm} -hidm {hidm} -mw {mw} -th {th} -mpt 10 -cm fofW --output-directory {out_dir} --write-clusters")

print(f"Processed all filterbanks in directory: {in_dir}")
