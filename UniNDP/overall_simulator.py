from hnsw_sim import hnsw
from idx_widgets.dataset_manage import get_data,get_dist_type
from idx_widgets.dataset_manage import gen_idx_name,get_dataset_idx_location
from mapping.mapping_manager import get_mapping
import time
import pickle
from  idx_widgets.tools import cal_recall
import numpy as np
import argparse
import csv
import os
from idx_widgets.tools import load_fvecs
from sim import sim
import logging

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
        "--idx_type","-idxt",
        type=str,
        default="hnsw",
        help="Index type, only support hnsw/cagra (default: %(default)s)"
    )
    parser.add_argument(
        "--query_num",
        type=int,
        default=36,
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

    if high_performance:
        ddr_cfg_path="../python/ddr_cfg/myddr5_hp.yaml"
    else:
        ddr_cfg_path="../python/ddr_cfg/myddr5.yaml"

    for dataset_name, ef_search in zip(["sift1M", "gist1M", "glove1_2M", "wiki1M", "msmacro8M", "sift100M"], [35, 60, 1000, 70, 35, 85]):
        
        if dataset_name != "glove1_2M":
            significance = 0.05
        else:
            significance = 0.005

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
        if do_sampling and not idx_type.endswith("_pca"):
            idx_type = f"{idx_type}_pca"
        idx_path=gen_idx_name(dataset_name, idx_type)

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
            if not os.path.exists(f"../python/mapping_cfg/{dataset_name}.npy"):
                cache_mapping = hnsw_idx.cache_mapping()
                arr = np.array(cache_mapping, dtype=object)
                np.save(f"../python/mapping_cfg/{dataset_name}.npy", arr, allow_pickle=True)     
            else:
                loaded_arr = np.load(f"../python/mapping_cfg/{dataset_name}.npy", allow_pickle=True)
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

        ####### Conduct Search #######
        print("Conducting search...")
        # 这次搜索使用的维度数
        used_dim_sum = 0
        exit_dims = [0 for _ in range(dim+1)]
        hnsw_idx.clear_states()

        # hnsw_idx.cache_mapping()

        hnsw_idx.ef = max(ef_search, k)
        
        # ids存的是每个查询的k个近邻的id
        ids = np.zeros((seleceted_query_num, k), dtype=int)
        vec_dist_cal_trace = []
        for i in range(seleceted_query_num):
            hnsw_idx.bfs_round = 0
            result, used_dim_sum_this_query, exit_dims_this_query, vec_dist_cal_trace_this_query = hnsw_idx.searchKnn(query[i] ,k=k)
            vec_dist_cal_trace.extend(vec_dist_cal_trace_this_query)

            used_dim_sum += used_dim_sum_this_query
            for j in range(dim+1):
                exit_dims[j] += exit_dims_this_query[j]

            items = []
            while not result.empty():
                items.append(result.get())

            assert len(items) == k
            for j, (dist, eid) in enumerate(items):
                ids[i, j] = eid
            print(f"solved {i} queries", end='\r')

        ######## Calculation Recall ########
        recall=cal_recall(ids, gt[:seleceted_query_num], k=k)
        print()
        print(f"Recall: {recall:.4f}")

        if not high_performance:
            tot = 0
            exit_freq = [0 for _ in range(dim+1)]
            for j in range(1, dim+1):
                tot += exit_dims[j]
            for j in range(1, dim+1):
                exit_freq[j] = exit_dims[j] / tot if tot != 0 else 0

            early_stop_dir = os.path.join(os.path.dirname(__file__), "..", "result", "early_stop_freq")
            os.makedirs(early_stop_dir, exist_ok=True)
            np.save(os.path.join(early_stop_dir, f"{dataset_name}.npy"), exit_freq)

        ######## Gen Command Trace ########
        cmd_trace, CPU_lat, cache_hit_rate = ddr_mapping.gen_cmd_trace(
                                vec_idx_trace=vec_dist_cal_trace,
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
                    "recall",
                    "qps",
                ],
            )
            writer.writeheader()
            writer.writerows(result_rows)

    print(f"Saved all-dataset QPS results to {csv_path}")
