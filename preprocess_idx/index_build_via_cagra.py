import cupy as cp
from cuvs.neighbors import cagra, hnsw
import os,time
import numpy as np
from idx_tools.idx_functions import cupy_load,PCA_gpu,save_fvecs
from idx_tools.dataset_manage import gen_idx_name,get_data,get_dataset_info,get_dist_type,get_dataset_idx_location
import torch
import argparse
import math

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def green_print(*args, **kwargs):
    print(GREEN + " ".join(str(x) for x in args) + RESET, **kwargs)


def red_print(*args, **kwargs):
    print(RED + " ".join(str(x) for x in args) + RESET, **kwargs)


def _as_float32_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32)


##### Varience calculation #####
def get_dx_output_prefix(dataset_name):
    filename = f"{dataset_name}"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "result", "Varience")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

def save_dx_from_dp_sqr(dp_sqr, output_prefix):
    # Strictly follow idx_widgets/get_dx.py for all distance types.
    n_s, dim = dp_sqr.shape
    dx = np.zeros((dim))
    dt_sqr = dp_sqr[:, dim - 1]
    dt_sqr = np.where(np.abs(dt_sqr) < 1e-12, 1e-12, dt_sqr)
    for d in range(dim):
        for i in range(n_s):
            dx[d] += (dp_sqr[i, d] / dt_sqr[i] - 1) ** 2
    dx /= n_s
    np.save(output_prefix, dx)
    green_print(f"Dx saved to {output_prefix}.npy, shape={dx.shape}")

def build_get_dx_dp_sqr(xp0, xp1, cdf_lmd):
    # Rebuild a dedicated DP_SQR for Dx with the exact get_dx.py formula.
    n_s, dim = xp0.shape
    dp_sqr = np.zeros((n_s, dim))
    sum_p = np.zeros((n_s))
    total_lmd = cdf_lmd[dim - 1]
    for d in range(dim):
        diff = xp0[:, d] - xp1[:, d]
        sum_p += diff * diff
        dp_sqr[:, d] = sum_p * total_lmd / cdf_lmd[d]
    return dp_sqr

def get_ip_normalized_residual(dp_sqr, norms):
    dt_sqr = dp_sqr[:, -1]
    safe_norms = np.where(np.abs(norms) < 1e-12, 1e-12, norms)
    return (dt_sqr[:, np.newaxis] - dp_sqr) / safe_norms


# Return generated index file paths.
def create_index(dataset_name,graph_degree,ef_construction,disable_cagra_to_cpu=False,cover=False,
                    do_sampling=False,gen_dx=False,AE_mode=True):
    type_name="cagra"
    if do_sampling:
        type_name="cagra_pca"
    gpu_idx_path=gen_idx_name(dataset_name, type_name, type="GPU",graph_degree=graph_degree,ef_construction=ef_construction)
    cpu_idx_path=gen_idx_name(dataset_name, type_name, type="CPU",graph_degree=graph_degree,ef_construction=ef_construction)

    gpu_idx_exist = os.path.exists(gpu_idx_path) if not AE_mode else True
    cpu_idx_exist = os.path.exists(cpu_idx_path)

    if cover==False:
        if gpu_idx_exist and cpu_idx_exist and not do_sampling:
            green_print(f"Index existed! It is {gpu_idx_path}; {cpu_idx_path}")
            return gpu_idx_path,cpu_idx_path

    green_print(f"Make sure the index path is correct: {cpu_idx_path}")
    
    data,dtype = get_data(dataset_name, db_data=True)
    N, D = data.shape
    dist_type= get_dist_type(dataset_name)

    if do_sampling:
        start = time.time()
        print("Do PCA sampling...")
        N_s = 100000
        # Probability range used for epsilon preprocessing.
        significances = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        # P is the projection matrix (eigenvectors), LMD is the eigenvalue set.
        P_np, LMD_np = PCA_gpu(data.T)  # Returned P and LMD are CPU-side numpy float32 arrays.
        data_tensor = _as_float32_tensor(data)  # Convert database vectors to torch.Tensor.
        P_np = _as_float32_tensor(P_np)  # Convert P to torch.Tensor.
        dataP_tensor = torch.matmul(data_tensor, P_np)
        # Convert to NumPy for downstream operations.
        data = dataP_tensor.numpy()
        # Save intermediate results.
        save_fvecs(f"{get_dataset_idx_location(dataset_name)}/P.fvecs", P_np)
        save_fvecs(f"{get_dataset_idx_location(dataset_name)}/LMD.fvecs", LMD_np)
        os.makedirs(f"{get_dataset_idx_location(dataset_name)}/E/", exist_ok=True)
        # Compute cumulative eigenvalue sum; this approximates the expected squared norm in reduced dimensions.
        CDF_LMD = np.cumsum(LMD_np)
        # Randomly sample N_s pairs of vectors from the database.
        pairs = np.random.randint(0, N, (2, N_s))
        # Gather vectors according to sampled pairs.
        XP0 = data[pairs[0]]
        XP1 = data[pairs[1]]
        # Create an N_s x D matrix to store estimated distances per used dimension.
        DP_SQR = np.zeros((N_s, D))
        # Store running true distances.
        sumP = np.zeros((N_s))
        NORM = np.zeros((N_s))
        NORMs = np.zeros((N_s, D))
        DX_DP_SQR = None
        if gen_dx:
            dx_path_prefix = get_dx_output_prefix(dataset_name)
            # Dx is computed independently from E and always follows get_dx.py.
            DX_DP_SQR = build_get_dx_dp_sqr(XP0, XP1, CDF_LMD)
        if dist_type == "l2":
            # Preprocess epsilon under different acceptance probabilities.
            for d in range(D):
                sumP += (XP0[:, d] - XP1[:, d]) * (XP0[:, d] - XP1[:, d])
                DP_SQR[:, d] = sumP * CDF_LMD[D - 1] / CDF_LMD[d]

            for significance in significances:
                DT_SQR = DP_SQR[:, D - 1] + 1e-12
                DP_SQR_DIF = np.sort(DP_SQR / DT_SQR[:, np.newaxis] - 1, axis=0)[::-1]
                EP = DP_SQR_DIF[int(N_s * significance),:][np.newaxis, :]
                save_fvecs(f"{get_dataset_idx_location(dataset_name)}/E/{significance}.fvecs", EP)

            if gen_dx:
                save_dx_from_dp_sqr(DX_DP_SQR, dx_path_prefix)
        else:
            for d in range(D):
                sumP += XP0[:, d] * XP1[:, d]
                DP_SQR[:, d] = sumP 
            
            for i in range(N_s):
                NORM[i] = math.sqrt(np.dot(XP0[i,:], XP0[i,:])) * math.sqrt(np.dot(XP1[i,:], XP1[i,:]))
                for d in range(D):
                    NORMs[i, d] = NORM[i] * (CDF_LMD[D - 1] - CDF_LMD[d]) / CDF_LMD[D - 1]
        
            ip_residual = get_ip_normalized_residual(DP_SQR, NORMs)
            for significance in significances:
                DP_SQR_DIF = np.sort(ip_residual, axis=0)[::-1]
                EP = DP_SQR_DIF[int(N_s * significance),:][np.newaxis, :]
                save_fvecs(f"{get_dataset_idx_location(dataset_name)}/E/{significance}.fvecs", EP)
            if gen_dx:
                save_dx_from_dp_sqr(DX_DP_SQR, dx_path_prefix)

        green_print(f"PCA sampling sucess!")

    if cpu_idx_exist:
        green_print(f"Index existed! It is {cpu_idx_path}")
        return gpu_idx_path,cpu_idx_path

    print("Loading data to GPU VRAM for CAGRA index building...")

    if dtype == "fp32":
        vecs=cupy_load(data, cp.float32, False)
    elif dtype == "uint8":
        vecs=cupy_load(data, cp.uint8, False)
    elif dtype == "int8":
        vecs=cupy_load(data, cp.int8, False)
    else:
        red_print(f"Unsupported data type: {dtype}")
        raise ValueError(f"Unsupported data type: {dtype}")


    metric  = "sqeuclidean" if dist_type=="l2" else "inner_product"
    green_print(f"Data is loaded! Data shape = {data.shape}, dtype = {data.dtype}. Distance type: {dist_type}, metric: {metric}")
    cagra_params = cagra.IndexParams(
        metric=metric,
        graph_degree=graph_degree,              # Final graph degree d.
        intermediate_graph_degree=graph_degree*2 # Initial graph degree d_init.
    )

    # Begin index building
    green_print(f"Building CAGRA index with params graph_degree = {graph_degree}, intermediate_graph_degree = {graph_degree*2}....")
    cagra_index = cagra.build(cagra_params, vecs)

    print(f"AE mode, skip saving GPU index.")
    if not os.path.exists(gpu_idx_path) and not AE_mode:
        os.makedirs(os.path.dirname(gpu_idx_path), exist_ok=True)
        cagra.save(gpu_idx_path, cagra_index, include_dataset=True)
        green_print(f"Index created and saved to {gpu_idx_path}")

    print(f"Transfer to CPU index...")
    if (not os.path.exists(cpu_idx_path) )and (not disable_cagra_to_cpu):
        os.makedirs(os.path.dirname(cpu_idx_path), exist_ok=True)
        hnsw_params = hnsw.IndexParams(
            hierarchy="cpu",  # "none" = single-layer; "cpu" = CPU-assisted multi-layer.
            ef_construction=ef_construction
        )
        hnsw_index = hnsw.from_cagra(hnsw_params, cagra_index)
        # Save the index for CPU
        print(f"Saving CPU index...")
        hnsw.save(cpu_idx_path, hnsw_index)
    
        green_print(f"Index created and saved to {cpu_idx_path}")

    return gpu_idx_path, cpu_idx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ANNS search with configurable parameters"
    )
    parser.add_argument(
        "--dataset_name", "-d",
        type=str,
        default="sift1M",
        help="Name of the dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--graph_degree", "-gd",
        type=int,
        default=64,
        help="Graph construction degree (default: %(default)s)"
    )
    parser.add_argument(
        "--ef_construction", "-ef",
        type=int,
        default=200,
        help="Construction parameter for trsnifering CAGRA to HNSW (default: %(default)s). 100 is good for SIFT1B."
    )
    parser.add_argument(
        "--no_cagra_to_cpu", "-nocpu",
        action="store_true",
        help="Construction parameter for converting CAGRA to HNSW on CPU. Default is True in case that you regret."
    )
    parser.add_argument(
        "-gpu_id",
        type=int,
        default=0,
        help="Which GPU to use (default: %(default)s)"
    )
    parser.add_argument(
        "-sampling", "-s",
        action="store_true",
        default=False,
        help="Use the sampling or not"
    )
    parser.add_argument(
        "-gen_dx",
        action="store_true",
        default=False,
        help="Generate Dx npy file (requires sampling and l2 distance)."
    )
    parser.add_argument(
        "-cover",
        action="store_true",
        default=False,
        help="Cover the built index or not (default: %(default)s)"
    )

    green_print(f"PID={os.getpid()}")

    args = parser.parse_args()
    
    dataset_name=args.dataset_name
    graph_degree=args.graph_degree
    ef_construction=args.ef_construction
    gpu_id= args.gpu_id
    disable_cagra_to_cpu = args.no_cagra_to_cpu
    # Sampling
    do_sampling = args.sampling
    # Cover index
    cover = args.cover

    # Index building
    cp.cuda.Device(gpu_id).use()
    begin = time.time()
    print(
        f"Building index for dataset {dataset_name} with graph degree {graph_degree} and ef construction {ef_construction} on GPU {gpu_id}",
    )
    gpu_idx_path, cpu_idx_path = create_index(
        dataset_name,
        graph_degree,
        ef_construction,
        disable_cagra_to_cpu=disable_cagra_to_cpu,
        do_sampling=do_sampling,
        cover=cover,
        gen_dx=args.gen_dx,
    )
    end = time.time()
    green_print(f"Finish building index! Time cost: {end - begin:.2f} s")