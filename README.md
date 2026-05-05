# AE Process Instruction

We provide two options for Artifact Evaluation.

1. Download the pre-built index, then run our scripts to start the simulation. This can reproduce the same results reported in the paper, and require only CPU and fewer hardware resource.
2. Build the index from scratch. This requires a GPU and large amounts of VRAM and DRAM, and costs much time.

We recommend using the pre-built indexes for simulation, as they can reproduce the same results reported in the paper.
If indexes are built from scratch, graph construction in cuVS involves randomness, the result will be slighly different.

## Option 1: Use Pre-built Index to Reproduce Results

## Hardware requirements:

A server with an x86 processor, at least 16 CPU cores, and at least 256\,GB of DRAM. 
More resource will accelerate the simulation.
You can use the '-j' option in bash scripts to adjust the cores used in parallel simulation.

## Software:

```bash
conda create -n naszip python=3.12
bash init_env.sh
pip install torch torchvision
pip install huggingface_hub matplotlib pandas seaborn numba
```

## 1. Prepare Datasets:

SIFT, GIST, GloVe, and Wiki require the original datasets:
```bash
python download_full_dataset.py SIFT GIST GloVe Wiki --output-dir Datasets
```
Other datasets only require queries and ground truth:
```bash
python download_query_groundtruth.py
```

## 2. Prepare Pre-built Index:

```bash
python download_anns_idx_cache.py
```

## 3. Simulation and Plot Procedure

The workflow is as follows: run the bash scripts to execute the NDP simulation and generate NASZIP results; use sync_csv_results.py to update these results into result/Data (baseline data are pre-filled, while NASZIP results are written into the CSV files); and finally generate figures using the plotting scripts based on the CSV files.

### 3.1 Figure 8
About 10 minutes.
```bash
bash fee_dim_freq.sh
bash get_var.sh
cd result && python sync_csv_results.py
cd Plot && python plot_fig_8.py && cd ..
```

### 3.2 Figure 15 & 18
About 1 h.
```bash
bash overall_basic.sh
cd result && python sync_csv_results.py
cd Plot && python plot_fig_15.py && cd ..
cd Plot && python plot_fig_18.py && cd ..
```

### 3.3 Figure 16
About 2.5 h.
```bash
bash overall_hp.sh
cd result && python sync_csv_results.py
cd Plot && python plot_fig_16.py && cd ..
```

### 3.4 Figure 19
About 2.5 h.
```bash
bash qps_vs_recall_SIFT.sh
bash qps_vs_recall_GloVe.sh
cd result && python sync_csv_results.py
cd Plot && python plot_fig_19.py && cd ..
```

### 3.5 Figure 22
About 1 h.
```bash
bash prefetch_hit_rate.sh
bash cache_hit_rate.sh
cd result && python sync_csv_results.py
cd Plot && python plot_fig_22.py && cd ..
```

# Option 2: Process for Building Index from Scratch

## Hardware requirements:

If you build the 100M BigANN index, 80 GB VRAM is required.
If you do not build the 100M BigANN index, 24 GB VRAM is sufficient.
Use as many CPU cores as possible.
Use as much DRAM as possible (if you build 100M BigANN, at least 320 GB).

## Software Requirements:

```bash
conda create -n naszip python=3.12
bash init_env.sh
conda install -c rapidsai -c conda-forge cuvs=25.06.00 cuda-version=12
pip install cupy-cuda12x==12.3.0
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
pip install cuda-python==11.8.5 cuda-bindings==12.9.4
pip install huggingface_hub matplotlib pandas seaborn numba
```

## Download datasets:

```bash
python download_full_dataset.py SIFT GIST GloVe BigANN MS_MARCO Wiki --output-dir Datasets
```

## Generate Index (e.g., `0` is the GPU ID):

Note: Building the BigANN index requires significant time and resources.  
If you want to build it:
```bash
bash build_index.sh 0
```

If you want to skip BigANN index construction, download the pre-built index (BigANN only):
```bash
python download_anns_idx_cache.py --dataset bigann100m
bash build_index.sh 0
```

# Others
UniNDP is modified to support rank level parallelism.

