import argparse
import math
import os
import time

import numpy as np
import torch

from idx_tools.dataset_manage import get_data, get_dist_type
from idx_tools.idx_functions import PCA_cpu, PCA_gpu


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


def get_dx_output_prefix(dataset_name):
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "result", "Varience")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, dataset_name)


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


def _has_usable_gpu():
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _compute_pca(data, use_gpu="auto"):
    if use_gpu not in {"auto", "true", "false"}:
        raise ValueError(f"Invalid use_gpu option: {use_gpu}")

    if use_gpu == "true":
        if not _has_usable_gpu():
            raise RuntimeError("GPU was requested, but no usable GPU was detected.")
        green_print("GPU detected, using PCA_gpu.")
        return PCA_gpu(data.T)

    if use_gpu == "auto" and _has_usable_gpu():
        green_print("GPU detected, using PCA_gpu.")
        return PCA_gpu(data.T)

    if use_gpu == "false":
        red_print("GPU disabled by option, using PCA_cpu.")
    else:
        red_print("No GPU detected, falling back to PCA_cpu.")
    return PCA_cpu(data.T)


def generate_variance(dataset_name, use_gpu="auto"):
    data, _ = get_data(dataset_name, db_data=True)
    n, dim = data.shape
    dist_type = get_dist_type(dataset_name)

    start = time.time()
    print("Do PCA sampling...")

    n_s = 100000
    p_np, lmd_np = _compute_pca(data, use_gpu=use_gpu)
    data_tensor = _as_float32_tensor(data)
    p_tensor = _as_float32_tensor(p_np)
    data = torch.matmul(data_tensor, p_tensor).numpy()

    cdf_lmd = np.cumsum(lmd_np)
    pairs = np.random.randint(0, n, (2, n_s))
    xp0 = data[pairs[0]]
    xp1 = data[pairs[1]]

    if dist_type != "l2":
        raise ValueError("Variance output is only supported for l2 datasets.")

    dx_path_prefix = get_dx_output_prefix(dataset_name)
    dx_dp_sqr = build_get_dx_dp_sqr(xp0, xp1, cdf_lmd)
    save_dx_from_dp_sqr(dx_dp_sqr, dx_path_prefix)

    green_print(f"PCA sampling success! Time cost: {time.time() - start:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate variance-related PCA sampling artifacts"
    )
    parser.add_argument(
        "--dataset_name", "-d",
        type=str,
        default="sift1M",
        help="Name of the dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--use_gpu",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to use GPU for PCA: auto, true, or false (default: %(default)s)"
    )

    green_print(f"PID={os.getpid()}")

    args = parser.parse_args()

    begin = time.time()
    print(f"Generating variance artifacts for dataset {args.dataset_name}")
    generate_variance(
        args.dataset_name,
        use_gpu=args.use_gpu,
    )
    green_print(f"Finished. Time cost: {time.time() - begin:.2f} s")
