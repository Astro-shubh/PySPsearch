# PySPsearch

A high-performance Python and C++ accelerated pulsar and fast radio burst (FRB) transient search pipeline. It features optimized Fast Dispersion Measure Transform (FDMT) support, modular chunk-based sliding-window processing, a custom high-performance C++ Boost R-tree spatial clustering engine (`fofW`), and DBSCAN verification.

---

## Installation

To install and set up PySPsearch for development or production use, follow these steps:

### 1. Clone the Repository
Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/Astro-shubh/PySPsearch.git
cd PySPsearch
```

### 2. Create and Activate the Conda Environment
Create the conda environment using the provided dependency configuration file:
```bash
conda env create -f environment.yml
conda activate pyspsearch_env  # (Or your custom environment name)
```

### 3. Build C++ Extensions In-Place
Compile the custom C++ backend modules (such as the R-tree clustering extensions) in-place:
```bash
python setup.py build_ext --inplace
```

---

## Usage

Run the pipeline using the command-line interface:
```bash
python pipeline.py input_file.fil --lodm 0.0 --hidm 100.0 --threshold 6.0 --write-clusters --output-directory ./clusters_output
```
