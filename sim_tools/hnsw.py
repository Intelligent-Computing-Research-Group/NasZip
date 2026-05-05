import struct, numpy as np
import queue
import os
import multiprocessing as mp
from idx_tools.idx_functions import load_fvecs
from PCA import PCA
import logging
import tqdm

from Dfloat import Dfloat
from UniNDP.tools.share import SimConfig


_CACHE_MAPPING_HNSW = None
_CACHE_MAPPING_MAX_WORKERS = 24


def _init_cache_mapping_worker(hnsw_obj):
    global _CACHE_MAPPING_HNSW
    _CACHE_MAPPING_HNSW = hnsw_obj


def _compute_cache_mapping_stats(i):
    rank = [0 for _ in range(_CACHE_MAPPING_HNSW.ra)]
    for nbr in _CACHE_MAPPING_HNSW.first_levels_nbr[i]:
        rank[nbr % _CACHE_MAPPING_HNSW.ra] += 1
    local_hist = {}
    max_rank = 0
    for count in rank:
        if count > max_rank:
            max_rank = count
        local_hist[count] = local_hist.get(count, 0) + 1
    return i, rank[0], max_rank, local_hist


def _compute_cache_mapping_stats_chunk(start_end):
    start, end = start_end
    return [_compute_cache_mapping_stats(i) for i in range(start, end)]

# Only used for searching
class HierarchicalNSW:
    def __init__(self, dist_type="l2",dim=128,ef=16,
                 dataset_name=None,
                 high_performance=None,
                 index_path=None,
                 do_dfloat=False,
                 sampling_info=None):
        self.index_path=index_path
        self.dist_type = dist_type
        self.dim = dim
        self.ef = ef
        self.do_dfloat = do_dfloat

        if high_performance:
            cfg_path = "../sim_tools/ddr_cfg/myddr5_hp.yaml"
        else:
            cfg_path = "../sim_tools/ddr_cfg/myddr5.yaml"
        SimConfig.read_from_yaml(cfg_path)

        self.ra = SimConfig.ch * SimConfig.ra

        #### Performance counters ###
        # Count how many BFS rounds are executed.
        self.bfs_round=0
        # Count how many vector distance computations are performed.
        self.dist_cal_cnt = 0    

        # Number of dimensions used for visited points.
        self.used_dims = []

        # Sampling Info: Is a dict contains:
        # - sampling: "type": "PCA", "L_path": xxx, "E_path": xxx 
        self.sampling_info = sampling_info

        # PCA sampling data to use
        self.using_PCA=False
        if self.sampling_info is not None:
            if self.sampling_info["type"] == "PCA":
                self.PCA=PCA(self.dim, self.sampling_info, self.dist_type, do_dfloat, dataset_name)
                self.using_PCA=True
        
        # Load Index
        self.load_index(self.index_path)

        # Visited set.
        self.visited = [False for _ in range(self.cur_element_count)]

        # List of visited points whose distances were computed.
        self.vec_dist_cal_trace = []

        self.changed = []
        self.success_num = []
        self.top = []
        self.updated_top = []

    # Used to repeat searches with different parameters.
    def clear_states(self):
        self.vec_dist_cal_trace = []
        self.used_dims = []
        self.changed = []
        self.success_num = []
        self.top = []
        self.updated_top = []
        self.visited = [False for _ in range(self.cur_element_count)]
        self.dist_cal_cnt = 0
        self.bfs_round = 0

    def set_hw_mapping(self,hw_map):
        self.ddr_mapping = hw_map

    def init_visited(self):
        self.visited = [False for _ in range(self.cur_element_count)]

    # Load hnsw format index.
    def load_index(self, index_path, log_level=logging.INFO):
        print(f"Loading index from {index_path}")
        logging.basicConfig(level=log_level)
        with open(index_path, 'rb') as f:

            fmt8f = f'<d'
            fmt4f = f'<f'
            fmt8i = f'<Q'
            fmt4i = f'<I'

            # offsetLevel0_
            v = struct.unpack(fmt8i, f.read(8))
            self.offsetLevel0_ = v[0]
            logging.debug(f"offsetLevel0_: {v[0]}")
            # max_elements_
            v = struct.unpack(fmt8i, f.read(8))
            self.max_elements_ = v[0]
            logging.debug(f"max_elements_,{v[0]}")
            # cur_element_count
            v = struct.unpack(fmt8i, f.read(8))
            self.cur_element_count = v[0]
            logging.debug(f"cur_element_count,{v[0]}")

            # size_data_per_element_
            v = struct.unpack(fmt8i, f.read(8))
            self.size_data_per_element_ = v[0]
            logging.debug(f"size_data_per_element_,{v[0]}")
            # label_offset_
            v = struct.unpack(fmt8i, f.read(8))
            self.label_offset_ = v[0]
            logging.debug(f"label_offset_,{v[0]}")
            # offsetData_
            v = struct.unpack(fmt8i, f.read(8))
            self.offsetData_ = v[0]
            logging.debug(f"offsetData_,{v[0]}")
            # maxlevel_
            v = struct.unpack(fmt4i, f.read(4))
            self.maxlevel_ = v[0]
            logging.debug(f"maxlevel_,{v[0]}")
            # enterpoint_node_
            v = struct.unpack(fmt4i, f.read(4))
            self.enterpoint_node_ = v[0]
            logging.debug(f"enterpoint_node_,{v[0]}")

            # maxM_
            v = struct.unpack(fmt8i, f.read(8))
            self.maxM_ = v[0]
            logging.debug(f"maxM_,{v[0]}")

            # maxM0_
            v = struct.unpack(fmt8i, f.read(8))
            self.maxM0_ = v[0]
            logging.debug(f"maxM0_,{v[0]}")

            # M_
            v = struct.unpack(fmt8i, f.read(8))
            self.M_ = v[0]
            logging.debug(f"M_,{v[0]}")

            # mult_
            v = struct.unpack(fmt8f, f.read(8))
            self.mult_ = v[0]
            logging.debug(f"mult_,{v[0]}")

            # efConstruction_
            v = struct.unpack(fmt8i, f.read(8))
            self.efConstruction_ = v[0]
            logging.debug(f"ef_construction_={v[0]}")

            # Compute size-related parameters.
            size_bytes_links_per_element_ = self.maxM_ * 4 + 4
            logging.debug(f"size_links_per_element_ = {size_bytes_links_per_element_} Bytes")

            # Data layout offsets.
            size_link_level0_ = self.maxM0_ * 4 + 4
            logging.debug(f"size_link_level0_={size_link_level0_}")

            self.first_levels_nbr = np.empty(self.cur_element_count, dtype=object)
            self.vec_data = np.empty((self.cur_element_count, self.dim), dtype=np.float32)
            self.data_ext_label = np.empty(self.cur_element_count, dtype=np.uint64)
            # print(self.size_data_per_element_)
            for i in range(self.cur_element_count):
                # Read one element record.
                raw = f.read(self.size_data_per_element_)

                nbr_num = struct.unpack(fmt4i, raw[0:4])[0]

                nbr_part = np.frombuffer(raw[4:4+nbr_num*4], dtype=np.uint32)
                self.first_levels_nbr[i] = nbr_part
                
                # Read vector data.
                data_start = size_link_level0_
                data_end = data_start + self.dim * 4
                self.vec_data[i] = np.frombuffer(raw[data_start:data_end], dtype=np.float32)
                
                # Read label.
                label_start = size_link_level0_ + self.dim * 4
                self.data_ext_label[i] = np.frombuffer(raw[label_start:label_start+8], dtype=np.uint64)[0]

            # Upper Layers
            self.upper_levels_nbr = [None for _ in range(self.cur_element_count)] 
            self.vec_level = [None for _ in range(self.cur_element_count)] 
            for i in range(self.cur_element_count):
                (linkListSizeByte,) = struct.unpack(fmt4i, f.read(4))
                if linkListSizeByte != 0:
                    raw = f.read(linkListSizeByte)
                    tmp = [i[0] for i in struct.iter_unpack('<I', raw)]
                    nbr_num = tmp[0]
                    self.upper_levels_nbr[i] = tmp[1:nbr_num + 1]
                    self.vec_level[i] = int(linkListSizeByte / size_bytes_links_per_element_)
                else:
                    self.vec_level[i] = 0

            print("Index loaded successfully.")

    def getExternalLabel(self, internal_id):
        return self.data_ext_label[internal_id]
    
    def get_distacne_with_ID(self, query_vector, vec_ID):
        if self.dist_type == "l2":
            return self.l2_distance(query_vector, self.getVecDataByInternalId(vec_ID),do_dfloat=self.do_dfloat)
        else:
            return self.ip_distance(query_vector, self.getVecDataByInternalId(vec_ID),do_dfloat=self.do_dfloat)

    def get_distance_with_ID_sampling(self, query_vector, vec_ID, lowerBound):
        if self.dist_type == "l2":
            return self.PCA.l2_distance_sampling(query_vector, 
                                                 self.getVecDataByInternalId(vec_ID), 
                                                 lowerBound,
                                                 do_dfloat=self.do_dfloat)
        else:
            return self.PCA.ip_distance_sampling(query_vector,
                                               self.getVecDataByInternalId(vec_ID),
                                               lowerBound,
                                               do_dfloat=self.do_dfloat)

    def ip_distance(self, vec1, vec2, do_dfloat=False):
        return -np.dot(np.array(vec1), np.array(vec2))

    def l2_distance(self, vec1, vec2, do_dfloat=False):
        diff = np.array(vec1) - np.array(vec2)
        return np.dot(diff, diff)
    
    def getVecDataByInternalId(self, internal_id):
        return self.vec_data[internal_id]

    def searchKnn(self, query_vector, k=10):
        # Reset visited flags for each search.
        self.init_visited()

        # Stores (distance, internal_id).
        result = queue.PriorityQueue()

        # Entry point
        cur_node = self.enterpoint_node_
        min_distance = self.get_distacne_with_ID(query_vector, cur_node)
        self.vec_dist_cal_trace.append("LoadQuery")
        self.used_dims.append([0])
        self.changed.append(False)
        self.top.append(0)
        self.success_num.append([0 for _ in range(self.ra)])
        self.updated_top.append(0)

        self.vec_dist_cal_trace.append([cur_node])
        self.used_dims.append([self.dim])
        self.changed.append(True)
        self.top.append(cur_node)
        self.success_num.append([1] + [0 for _ in range(self.ra - 1)])
        self.updated_top.append(cur_node)

        # Search upper layers first.
        for level in range(self.maxlevel_, 0, -1):
            changed = True
            while changed:
                changed = False
                
            nbr_list = self.upper_levels_nbr[cur_node]

            self.bfs_round=+1

            dist_cal_node_list = []
            used_dims_list=[]
            success_num = [0 for _ in range(self.ra)]
            for nbr in nbr_list:
                # Compute distance.
                dist_cal_node_list.append(nbr)
                used_dims_list.append(self.dim)

                cur_distance = self.get_distacne_with_ID(query_vector, nbr)

                if cur_distance < min_distance:
                    # If a closer neighbor is found, use it as next entry point.
                    min_distance=cur_distance
                    cur_node = nbr
                    changed = True
                    success_num[nbr% self.ra] += 1
            self.vec_dist_cal_trace.append(dist_cal_node_list)
            self.used_dims.append(used_dims_list)
            self.success_num.append(success_num)
            if changed:
                self.changed.append(True)
            else:
                self.changed.append(False)
            self.top.append(cur_node)
            self.updated_top.append(cur_node)
        # Start from the selected entry point and search the base layer.
        # top_candidates is a max-priority heap implemented with negative distances.
        if self.using_PCA:
            if self.dist_type == "l2":
                top_candidates, used_dim_sum, exit_dims = self.search_base_layer_l2_sampling(cur_node, query_vector, min_distance, max(self.ef, k), k)
            else:
                top_candidates, used_dim_sum, exit_dims = self.search_base_layer_ip_sampling(cur_node, query_vector, min_distance, max(self.ef, k), k)
        else:
            top_candidates, used_dim_sum, exit_dims = self.search_base_layer(cur_node, query_vector, min_distance, max(self.ef, k))

        # Remove extra entries beyond top-k.
        while top_candidates.qsize() > k:
            top_candidates.get()

        # Convert internal IDs to external labels and push to result queue.
        while not top_candidates.empty():
            dist, internal_id = top_candidates.get()
            result.put((-dist, self.getExternalLabel(internal_id)))

        return result, used_dim_sum, exit_dims

    def search_base_layer(self, entry_id, query_vector, entry_query_dist,ef):
        used_dim_sum = 0

        # 2 Priority Queues
        top_candidates = queue.PriorityQueue()  # Max value on top (stored as negative distance).
        candidates = queue.PriorityQueue()      # Min value on top.

        # Initialize lowerBound with distance(entry, query).
        lowerBound=entry_query_dist
        top_candidates.put((-entry_query_dist, entry_id))
        candidates.put((entry_query_dist, entry_id))
        self.visited[entry_id] = True

        while not candidates.empty():
            # Pop the current best candidate (minimum distance).
            cur_node_pair = candidates.get()
            cur_node_dist,cur_node_id = cur_node_pair

            if cur_node_dist > lowerBound and top_candidates.qsize() == ef:
                break

            # Visit neighbors.
            cur_node_nbr_list = self.first_levels_nbr[cur_node_id]

            dist_cal_node_list = []
            used_dims_list = []
            self.bfs_round += 1
            success_num = [0 for _ in range(self.ra)]
            if not candidates.empty():
                last_best = candidates.queue[0]
            else:
                last_best = None

            updated_best = 0
            updated_best_dist = float("inf")

            for nbr in cur_node_nbr_list:
                # Skip if already visited.
                if self.visited[nbr]:
                    continue

                self.visited[nbr] = True

                # Compute distance.
                dist_cal_node_list.append(nbr)
                used_dims_list.append(self.dim)
                used_dim_sum += self.dim
                distance = self.get_distacne_with_ID(query_vector, nbr)

                if (top_candidates.qsize() < ef) or (distance < lowerBound):
                    candidates.put((distance, nbr))
                    top_candidates.put((-distance, nbr))
                    success_num[nbr % self.ra] += 1

                    if distance < updated_best_dist:
                        updated_best_dist = distance
                        updated_best = nbr

                # If full, drop the farthest candidate.
                if top_candidates.qsize() > ef:
                    top_candidates.get()

                # Update lowerBound.
                if not top_candidates.empty():
                    lowerBound = -top_candidates.queue[0][0]
            self.vec_dist_cal_trace.append(dist_cal_node_list)
            self.used_dims.append(used_dims_list)
            self.success_num.append(success_num)
            self.updated_top.append(updated_best if updated_best != 0 else 0)
            if not candidates.empty():
                if last_best != candidates.queue[0]:
                    self.changed.append(True)
                else:
                    self.changed.append(False)
                self.top.append(candidates.queue[0][1])
            else:
                self.changed.append(False)
                self.top.append(0)

        return top_candidates, used_dim_sum, [0 for _ in range(self.dim+1)]
    

    def search_base_layer_l2_sampling(self, entry_id, query_vector, entry_query_dist, ef, k):
        used_dim_sum = 0
        exit_dims = [0 for _ in range(self.dim+1)]
        
        answer = queue.PriorityQueue()
        top_candidates = queue.PriorityQueue()
        candidate_set = queue.PriorityQueue()

        lowerBound = entry_query_dist
        lowerBoundCan = entry_query_dist

        # answer is the final result set with size k.
        answer.put((-entry_query_dist, entry_id))
        # top_candidates is the candidate result set with size ef.
        top_candidates.put((-entry_query_dist, entry_id))
        # candidate_set stores candidates to be expanded.
        candidate_set.put((entry_query_dist, entry_id))

        self.visited[entry_id] = True

        while not candidate_set.empty():
            cur_node_pair = candidate_set.get()
            cur_node_dist, cur_node_id = cur_node_pair

            tmp_node_pair = top_candidates.get()    
            if cur_node_dist > -tmp_node_pair[0] and top_candidates.qsize() == ef - 1:
                top_candidates.put(tmp_node_pair)
                break
            top_candidates.put(tmp_node_pair)

            self.bfs_round += 1
            cur_node_nbr_list = self.first_levels_nbr[cur_node_id]
            dist_cal_node_list = []
            used_dims_list = []
            success_num = [0 for _ in range(self.ra)]

            updated_best_node = 0
            updated_best_dist = float("inf")

            if not candidate_set.empty():
                last_best = candidate_set.queue[0]
            else:
                last_best = None
            for nbr in cur_node_nbr_list:
                
                if self.visited[nbr]:
                    continue
            
                self.visited[nbr] = True
                if answer.qsize() < k:
                    dist_cal_node_list.append(nbr)
                    used_dims_list.append(self.dim)
                    used_dim_sum += self.dim
                    distance = self.get_distacne_with_ID(query_vector, nbr)
                    answer.put((-distance, nbr))
                    top_candidates.put((-distance, nbr))
                    candidate_set.put((distance, nbr))

                    if distance < updated_best_dist:
                        updated_best_dist = distance
                        updated_best_node = nbr

                    if not answer.empty():
                        tmp_node_pair = answer.get()
                        lowerBound = -tmp_node_pair[0]
                        answer.put(tmp_node_pair)
                    if not top_candidates.empty():
                        tmp_node_pair = top_candidates.get()
                        lowerBoundCan = -tmp_node_pair[0]
                        top_candidates.put(tmp_node_pair)
                    success_num[nbr % self.ra] += 1
                
                else:
                    distance, used_dim = self.get_distance_with_ID_sampling(query_vector, nbr, lowerBound)
                    dist_cal_node_list.append(nbr)
                    used_dims_list.append(used_dim)
                    used_dim_sum += used_dim
                    exit_dims[used_dim] += 1
                    if distance >= 0:
                        candidate_set.put((distance, nbr))
                        top_candidates.put((-distance, nbr))
                        answer.put((-distance, nbr))

                        if distance < updated_best_dist:
                            updated_best_dist = distance
                            updated_best_node = nbr

                        if top_candidates.qsize() > ef:
                            top_candidates.get()
                        if answer.qsize() > k:
                            answer.get()
                        if not answer.empty():
                            tmp_node_pair = answer.get()
                            lowerBound = -tmp_node_pair[0]
                            answer.put(tmp_node_pair)
                        if not top_candidates.empty():
                            tmp_node_pair = top_candidates.get()
                            lowerBoundCan = -tmp_node_pair[0]
                            top_candidates.put(tmp_node_pair)
                        success_num[nbr % self.ra] += 1
                    else:
                        if top_candidates.qsize() < ef or lowerBoundCan > -distance:
                            top_candidates.put((distance, nbr))
                            candidate_set.put((-distance, nbr))
                            success_num[nbr % self.ra] += 1
                        if top_candidates.qsize() > ef:
                            top_candidates.get()
                        if not top_candidates.empty():
                            tmp_node_pair = top_candidates.get()
                            lowerBoundCan = -tmp_node_pair[0]
                            top_candidates.put(tmp_node_pair)
            self.vec_dist_cal_trace.append(dist_cal_node_list)
            self.used_dims.append(used_dims_list)
            self.success_num.append(success_num)
            
            self.updated_top.append(updated_best_node if updated_best_dist < float("inf") else 0)
            
            if not candidate_set.empty():
                if last_best != candidate_set.queue[0]:
                    self.changed.append(True)
                else:
                    self.changed.append(False)
                self.top.append(candidate_set.queue[0][1])
            else:
                self.changed.append(False)
                self.top.append(0)

        return answer, used_dim_sum, exit_dims

    def search_base_layer_ip_sampling(self, entry_id, query_vector, entry_query_dist, ef, k):
        used_dim_sum = 0
        exit_dims = [0 for _ in range(self.dim+1)]

        # 2 Priority Queues
        top_candidates = queue.PriorityQueue()  # Max value on top (stored as negative distance).
        candidates = queue.PriorityQueue()      # Min value on top.

        # Initialize lowerBound with distance(entry, query).
        lowerBound=entry_query_dist
        top_candidates.put((-entry_query_dist, entry_id))
        candidates.put((entry_query_dist, entry_id))
        self.visited[entry_id] = True

        while not candidates.empty():
            # Pop the current best candidate (minimum distance).
            cur_node_pair = candidates.get()
            cur_node_dist,cur_node_id = cur_node_pair

            if cur_node_dist > lowerBound and top_candidates.qsize() == ef:
                break

            # Visit neighbors.
            cur_node_nbr_list = self.first_levels_nbr[cur_node_id]

            dist_cal_node_list = []
            used_dims_list = []
            self.bfs_round += 1
            success_num = [0 for _ in range(self.ra)]
           
            updated_best_node = 0
            updated_best_dist = float("inf")

            if not candidates.empty():
                last_best = candidates.queue[0]
            else:
                last_best = None
            for nbr in cur_node_nbr_list:
                # Skip if already visited.
                if self.visited[nbr]:
                    continue

                self.visited[nbr] = True

                # Compute distance.
                if top_candidates.qsize() < ef: 
                    dist_cal_node_list.append(nbr)
                    used_dims_list.append(self.dim)
                    used_dim_sum += self.dim
                    distance = self.get_distacne_with_ID(query_vector, nbr)
                    candidates.put((distance, nbr))
                    top_candidates.put((-distance, nbr))
                    if not top_candidates.empty():
                            tmp_node_pair = top_candidates.get()
                            lowerBound = -tmp_node_pair[0]
                            top_candidates.put(tmp_node_pair)
                    success_num[nbr % self.ra] += 1

                    if distance < updated_best_dist:
                        updated_best_dist = distance
                        updated_best_node = nbr

                else:
                    success, distance, used_dim = self.get_distance_with_ID_sampling(query_vector, nbr, lowerBound)
                    dist_cal_node_list.append(nbr)
                    used_dims_list.append(used_dim)
                    used_dim_sum += used_dim
                    exit_dims[used_dim] += 1
                    if success:
                        candidates.put((distance, nbr))
                        top_candidates.put((-distance, nbr))
                        if top_candidates.qsize() > ef:
                            top_candidates.get()
                        if not top_candidates.empty():
                            tmp_node_pair = top_candidates.get()
                            lowerBound = -tmp_node_pair[0]
                            top_candidates.put(tmp_node_pair)
                        success_num[nbr % self.ra] += 1
                        if distance < updated_best_dist:
                            updated_best_dist = distance
                            updated_best_node = nbr

            self.vec_dist_cal_trace.append(dist_cal_node_list)
            self.used_dims.append(used_dims_list)
            self.success_num.append(success_num)
            self.updated_top.append(updated_best_node if updated_best_dist < float("inf") else 0)

            if not candidates.empty():
                if last_best != candidates.queue[0]:
                    self.changed.append(True)
                else:
                    self.changed.append(False)
                self.top.append(candidates.queue[0][1])
            else:
                self.changed.append(False)
                self.top.append(0)

        return top_candidates, used_dim_sum, exit_dims

    # Build per-node neighbor-ID groupings based on interleaved data in each sub-channel.
    # Then merge those index lists, because each node can have a different neighbor-list length.
    def cache_mapping(self):
        cache_pack = [[] for _ in range(self.cur_element_count)]
        max_nbr_rank = [0 for _ in range(self.cur_element_count)]
        nbr_num_rank = [0 for _ in range(32)]
        remain = 16
        burst_num = 0
        waste = 0
        numm = 0
        pack = []
        apack = []
        worker_num = min(os.cpu_count() or 1, _CACHE_MAPPING_MAX_WORKERS, self.cur_element_count)
        stats = [None for _ in range(self.cur_element_count)]
        if os.name == "posix" and worker_num > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=worker_num, initializer=_init_cache_mapping_worker, initargs=(self,)) as pool:
                chunk_size = max(1, (self.cur_element_count + worker_num * 32 - 1) // (worker_num * 32))
                chunk_ranges = [
                    (start, min(start + chunk_size, self.cur_element_count))
                    for start in range(0, self.cur_element_count, chunk_size)
                ]
                stats_iter = pool.imap(_compute_cache_mapping_stats_chunk, chunk_ranges, chunksize=1)
                if tqdm is not None:
                    stats_iter = tqdm.tqdm(stats_iter, desc="Processing mapping of nbr lists:", total=len(chunk_ranges))
                for chunk_stats in stats_iter:
                    for i, rank0, max_rank, local_hist in chunk_stats:
                        stats[i] = (rank0, max_rank, local_hist)
        else:
            iter_range = range(self.cur_element_count)
            if tqdm is not None:
                iter_range = tqdm.tqdm(iter_range, desc="Processing mapping of nbr lists:", total=self.cur_element_count)
            for i in iter_range:
                rank = [0 for _ in range(self.ra)]
                for nbr in self.first_levels_nbr[i]:
                    rank[nbr % self.ra] += 1
                local_hist = {}
                max_rank = 0
                for count in rank:
                    if count > max_rank:
                        max_rank = count
                    local_hist[count] = local_hist.get(count, 0) + 1
                stats[i] = (rank[0], max_rank, local_hist)

        for i, (rank0, max_rank, local_hist) in enumerate(stats):
            max_nbr_rank[i] = max_rank
            for count, freq in local_hist.items():
                if count < len(nbr_num_rank):
                    nbr_num_rank[count] += freq
            numm += rank0
            if remain > rank0:
                remain -= rank0
                apack.append(i)
            elif rank0 > 16:
                remain = 16 - rank0 + remain
                burst_num += 1
                pack.append(apack)
                for elem in apack:
                    cache_pack[elem] = apack
                apack = [i]
            else:
                waste = 16 - remain
                remain = 16 - rank0
                burst_num += 1
                pack.append(apack)
                for elem in apack:
                    cache_pack[elem] = apack
                apack = [i]
        pack.append(apack)
        for elem in apack:
            cache_pack[elem] = apack
        return cache_pack
