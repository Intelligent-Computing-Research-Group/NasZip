from sim_tools import hnsw
from idx_tools.dataset_manage import get_data,get_dist_type
from idx_tools.dataset_manage import gen_idx_name,get_dataset_idx_location
from sim_tools.mapping_manager import get_mapping
import time
import pickle
from  idx_tools.idx_functions import cal_recall
import numpy as np
import argparse
import csv
import os
from idx_tools.idx_functions import load_fvecs
from sim import sim
import logging
import multiprocessing as mp
from tqdm import tqdm


_CMD_TRACE = None
_QUERY_HNSW = None
_QUERY_ARRAY = None
_QUERY_K = None


def _simulate_latency_slice(start_end):
    start, end = start_end
    latency = 0
    for i in range(start, end):
        latency += sim(_CMD_TRACE[i], silent=True)
    return latency


def _init_query_worker():
    global _QUERY_HNSW, _QUERY_ARRAY, _QUERY_K


def _run_query_slice(query_id):
    _QUERY_HNSW.clear_states()
    _QUERY_HNSW.bfs_round = 0
    result, used_dim_sum_this_query, exit_dims_this_query = _QUERY_HNSW.searchKnn(_QUERY_ARRAY[query_id], k=_QUERY_K)

    items = []
    while not result.empty():
        items.append(result.get())

    assert len(items) == _QUERY_K
    ids_row = np.zeros(_QUERY_K, dtype=int)
    for j, (_, eid) in enumerate(items):
        ids_row[j] = eid

    return (
        query_id,
        ids_row,
        used_dim_sum_this_query,
        exit_dims_this_query,
        _QUERY_HNSW.vec_dist_cal_trace,
        _QUERY_HNSW.used_dims,
        _QUERY_HNSW.changed,
        _QUERY_HNSW.success_num,
        _QUERY_HNSW.top,
        _QUERY_HNSW.updated_top,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ANNS search with configurable parameters"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: %(default)s)"
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
        "--idx_type","-idxt",
        type=str,
        default="hnsw",
        help="Index type, only support hnsw/cagra (default: %(default)s)"
    )
    parser.add_argument(
        "--query_num",
        type=int,
        default=1000,
        help="Number of queries to benchmark (default: %(default)s)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=4,
        help="Batch size for processing queries (default: %(default)s)"
    )
    parser.add_argument(
        "-sampling", "-s",
        action="store_true",
        default=False,
        help="Use the sampling or not"
    )
    parser.add_argument(
        "-do_dfloat", "-df",
        action="store_true",
        default=False,
        help=""
    )
    parser.add_argument(
        "--mapping_choice","-map",
        type=str,
        default="vec_in_rank_interleave",
        help="Index type, only support hnsw/cagra (default: %(default)s)"
    )
    parser.add_argument(
        "-baseline", "-ba",
        action="store_true",
        default=False,
        help=""
    )
    parser.add_argument(
        "-high_performance", "-hp",
        action="store_true",
        default=False,
        help=""
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=0,
        help="Number of CPU workers for simulation (0=auto, default: %(default)s)"
    )
    parser.add_argument(
        "--query_workers",
        type=int,
        default=1,
        help="Number of worker processes for query-level search parallelism (default: %(default)s)"
    )

    ######## Parameter #######
    args = parser.parse_args()
    result_rows = []
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    os.makedirs(result_dir, exist_ok=True)

    k = args.k
    do_sampling = args.sampling
    test_query_num = args.query_num
    idx_type = args.idx_type
    mapping_choice = args.mapping_choice
    query_batch_size = args.batch_size
    do_dfloat = args.do_dfloat
    baseline = args.baseline
    high_performance = args.high_performance
    workers = args.workers
    query_workers = args.query_workers

    if high_performance:
        ddr_cfg_path="../sim_tools/ddr_cfg/myddr5_hp.yaml"
        Ks = [1, 10]
    else:
        ddr_cfg_path="../sim_tools/ddr_cfg/myddr5.yaml"
        Ks = [10]

    begin_time = time.time()
    for dataset_name, ef_search, gd in zip(["SIFT", "GIST", "GloVe", "Wiki", "MS_MARCO", "BigANN100M"],
                                        [40, 120, 2000, 280, 40, 85], [32, 64, 16, 32, 32, 32]):
        
        significance = 0.05

        dist_type = get_dist_type(dataset_name)
        sampling_info=None
        if do_sampling:
            # Load P matrix
            P=load_fvecs(f"{get_dataset_idx_location(dataset_name)}/P.fvecs")
            print(f"Loaded P matrix with shape: {P.shape}")
            query,_ = get_data(dataset_name, query=True)
            print(f"Loaded query with shape: {query.shape}")
            queryP = np.dot(query, P)
            query = queryP
            sampling_info={"type":"PCA",
                        "L_path":f"{get_dataset_idx_location(dataset_name)}/LMD.fvecs",
                        "E_path":f"{get_dataset_idx_location(dataset_name)}/E/{significance}.fvecs"}
        else:
            query,_ = get_data(dataset_name, query=True)

        num_query,dim = query.shape
        seleceted_query_num=test_query_num if test_query_num<num_query else num_query

        ####### Load Ground Truth #######
        gt = get_data(dataset_name, gt=True)

        ####### Load Index #######
        current_idx_type = idx_type
        if do_sampling and not current_idx_type.endswith("_pca"):
            current_idx_type = f"{current_idx_type}_pca"
        idx_path=gen_idx_name(dataset_name, current_idx_type, graph_degree=gd, ef_construction=args.ef_construction)

        ####### Creat Search Config #######
        hnsw_idx = hnsw.HierarchicalNSW(index_path=idx_path,
                                        dataset_name=dataset_name,
                                        high_performance=high_performance,
                                        dist_type=dist_type,
                                        dim=dim,
                                        ef=64,
                                        do_dfloat=do_dfloat,
                                        sampling_info=sampling_info)

        cache_mapping = None
        if not baseline:
            mapping_cfg_file_name = f"{dataset_name}_gd{gd}_ef{args.ef_construction}.npy"
            if not os.path.exists(f"../sim_tools/mapping_cfg/{mapping_cfg_file_name}"):
                cache_mapping = hnsw_idx.cache_mapping()
                arr = np.array(cache_mapping, dtype=object)
                os.makedirs("../sim_tools/mapping_cfg", exist_ok=True)
                np.save(f"../sim_tools/mapping_cfg/{mapping_cfg_file_name}", arr, allow_pickle=True)
            else:
                loaded_arr = np.load(f"../sim_tools/mapping_cfg/{mapping_cfg_file_name}", allow_pickle=True)
                cache_mapping = loaded_arr.tolist()
        print("Mapping finished successfully.")

        ####### Creat Mapping Config #######
        ddr_mapping = get_mapping(
                                mapping_name=mapping_choice,
                                ddr_cfg_path=ddr_cfg_path,
                                data_num=hnsw_idx.cur_element_count, 
                                dim=hnsw_idx.dim,
                                do_dfloat=do_dfloat,
                                dataset_name=dataset_name,
                                cache_mapping=cache_mapping)
        for k in Ks:
            ####### Conduct Search #######
            print("Conducting search...")
            # Accumulated dimension usage across queries in this search run.
            used_dim_sum = 0
            exit_dims = [0 for _ in range(dim+1)]
            hnsw_idx.clear_states()
            if hasattr(ddr_mapping.cache, "clear"):
                ddr_mapping.cache.clear()
            else:
                ddr_mapping.cache = []

            hnsw_idx.ef = max(ef_search, k)
            
            # ids[i] stores the k nearest-neighbor IDs for query i.
            ids = np.zeros((seleceted_query_num, k), dtype=int)
            if workers <= 1:
                for i in tqdm(
                    range(seleceted_query_num),
                    desc="Queries",
                    unit="q",
                    leave=False,
                ):
                    hnsw_idx.bfs_round = 0
                    result, used_dim_sum_this_query, exit_dims_this_query = hnsw_idx.searchKnn(query[i], k=k)

                    used_dim_sum += used_dim_sum_this_query
                    for j in range(dim+1):
                        exit_dims[j] += exit_dims_this_query[j]

                    items = []
                    while not result.empty():
                        items.append(result.get())

                    assert len(items) == k
                    for j, (_, eid) in enumerate(items):
                        ids[i, j] = eid
            else:
                if os.name != "posix":
                    raise RuntimeError("Query-level parallelism currently requires a POSIX system with fork support.")

                _QUERY_HNSW = hnsw_idx
                _QUERY_ARRAY = query
                _QUERY_K = k

                ctx = mp.get_context("fork")
                with ctx.Pool(processes=workers, initializer=_init_query_worker) as pool:
                    worker_outputs = [None for _ in range(seleceted_query_num)]
                    for output in tqdm(
                        pool.imap_unordered(_run_query_slice, range(seleceted_query_num)),
                        total=seleceted_query_num,
                        desc="Queries",
                        unit="q",
                        leave=False,
                    ):
                        worker_outputs[output[0]] = output

                for i, ids_row, used_dim_sum_this_query, exit_dims_this_query, vec_trace, used_dims_trace, changed_trace, success_num_trace, top_trace, updated_top_trace in worker_outputs:
                    ids[i] = ids_row
                    used_dim_sum += used_dim_sum_this_query
                    for j in range(dim + 1):
                        exit_dims[j] += exit_dims_this_query[j]
                    hnsw_idx.vec_dist_cal_trace.extend(vec_trace)
                    hnsw_idx.used_dims.extend(used_dims_trace)
                    hnsw_idx.changed.extend(changed_trace)
                    hnsw_idx.success_num.extend(success_num_trace)
                    hnsw_idx.top.extend(top_trace)
                    hnsw_idx.updated_top.extend(updated_top_trace)

            ######## Calculation Recall ########
            recall=cal_recall(ids, gt[:seleceted_query_num], k=k)
            print()
            print(f"Recall: {recall:.4f}")

            ######## Gen Command Trace ########
            cmd_trace, CPU_lat, cache_hit_rate = ddr_mapping.gen_cmd_trace(
                                    vec_idx_trace=hnsw_idx.vec_dist_cal_trace,
                                    used_dims_trace=hnsw_idx.used_dims,
                                    success_num=hnsw_idx.success_num,
                                    changed=hnsw_idx.changed,
                                    top=hnsw_idx.top,
                                    updated_top=hnsw_idx.updated_top,
                                    batch_size=query_batch_size,
                                    baseline=baseline)
            
            print("Command trace generated.")

            ######## Conduct Simulation ########
            latency_cycle = 0
            workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
            workers = min(workers, len(cmd_trace)) if cmd_trace is not None else workers
            if workers > 1 and len(cmd_trace) > 1:
                ctx = mp.get_context("fork")
                _CMD_TRACE = cmd_trace
                chunk_size = (len(cmd_trace) + workers - 1) // workers
                slices = []
                for w in range(workers):
                    start = w * chunk_size
                    end = min((w + 1) * chunk_size, len(cmd_trace))
                    if start < end:
                        slices.append((start, end))
                with ctx.Pool(processes=workers) as pool:
                    latency_parts = pool.map(_simulate_latency_slice, slices)
                latency_cycle = sum(latency_parts)
            else:
                for cmd_for_sim in cmd_trace:
                    latency_cycle += sim(cmd_for_sim, silent=True)

            qps = 2400 * test_query_num * 1000 * 1000 / (latency_cycle + CPU_lat)
            print(f"QPS: {qps}")

            if not high_performance:
                result_rows.append({
                    "dataset_name": dataset_name,
                    "recall": recall,
                    "qps": qps,
                    "latency": 100000 / qps,
                    "Distance Calculation": ddr_mapping.mem_acc / (ddr_mapping.mem_acc + ddr_mapping.load_cpu + ddr_mapping.merge_cpu + ddr_mapping.fetch_nbr),
                    "Partial Result Processing": (ddr_mapping.load_cpu + ddr_mapping.merge_cpu) / (ddr_mapping.mem_acc + ddr_mapping.load_cpu + ddr_mapping.merge_cpu + ddr_mapping.fetch_nbr),
                    "Neighbor List Fetch": ddr_mapping.fetch_nbr / (ddr_mapping.mem_acc + ddr_mapping.load_cpu + ddr_mapping.merge_cpu + ddr_mapping.fetch_nbr),
                })
            else:
                result_rows.append({
                    "dataset_name": dataset_name,
                    "k":k,
                    "recall": recall,
                    "qps": qps,
                })

    if not high_performance:
        csv_path = os.path.join(result_dir, "NasZip_overall.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset_name",
                    "recall",
                    "qps",
                    "latency",
                    "Distance Calculation",
                    "Partial Result Processing",
                    "Neighbor List Fetch",
                ],
            )
            writer.writeheader()
            writer.writerows(result_rows)
    else:
        csv_path = os.path.join(result_dir, "NasZip_overall_hp.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset_name",
                    "k",
                    "recall",
                    "qps",
                ],
            )
            writer.writeheader()
            writer.writerows(result_rows)

    end_time = time.time()
    print(f"Time cost: {end_time - begin_time} seconds")
    print(f"Saved all-dataset QPS results to {csv_path}")
