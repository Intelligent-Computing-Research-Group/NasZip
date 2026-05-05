
from tools.share import SimConfig
import math
from idx_tools.idx_functions import *
from idx_tools.dataset_manage import df_cfg_path
import numpy as np
from tools import *
import yaml
from collections import deque

class VecInRankInterleaveMapping:
    def __init__(self,cfg_path,data_num,dim,do_dfloat,dataset_name,cache_mapping=None):
        cfg_path = cfg_path
        SimConfig.read_from_yaml(cfg_path)

        self.mem_acc = 0
        self.merge_cpu = 0
        self.load_cpu = 0
        self.fetch_nbr = 0

        self.data_num = data_num
        self.data_dim = dim
        self.data_bits = 32  # Data stored on PIM uses 32 bits.
        self.total_data_bytes = self.data_num * self.data_dim * (self.data_bits // 8)
        self.do_dfloat = do_dfloat
        self.cache_mapping = cache_mapping

        self.pf_num = [0 for _ in range(50000)]
        self.pf_hit = [0 for _ in range(50000)]

        self.cache_size = 1000
        # The cache stores one "burst pack" (list) returned by cache_mapping[node_id].
        # Keep an element frequency table to avoid linear scans on self.cache for hit checks.
        self.cache = deque()
        self.cache_elem_cnt = {}
        
        if do_dfloat:
            with open(df_cfg_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.col_size = []
            self.col_bound = []
            self.bits_type = data[dataset_name]['bits_type']
            self.bits = data[dataset_name]['bits']
            self.dims_bound = data[dataset_name]['dims']
            for i in range(self.bits_type):
                self.col_size.append(int(SimConfig.co_w  / self.bits[i]))
                if i >= 1 :
                    self.col_bound.append((self.dims_bound[i] - self.dims_bound[i - 1]) / self.col_size[i] + self.col_bound[i - 1])
                else:
                    self.col_bound.append(self.dims_bound[i] / self.col_size[i])

        print(f"Data number: {self.data_num}, Data dimension: {self.data_dim}, Data bits: {self.data_bits}")
        print(f"Total data bytes: {self.total_data_bytes} Bytes")

        self.hw_num_total_rank = SimConfig.ra * SimConfig.ch
        self.hw_total_bank_for_mapping = SimConfig.bg * SimConfig.ba * self.hw_num_total_rank
        self.hw_bytes_per_bank = SimConfig.ro * SimConfig.co*SimConfig.co_w/8
        self.hw_bytes_per_device = self.hw_bytes_per_bank*SimConfig.bg*SimConfig.ba

        self.do_mapping()

    def do_mapping(self):
        print("+++++++++++Generating mapping info...+++++++++++++")
        # Number of data bytes each rank can store.
        hw_bytes_per_rank = self.hw_bytes_per_device * SimConfig.de
        hw_bytes_total = hw_bytes_per_rank * self.hw_num_total_rank
        print(f"总共的内存大小= {hw_bytes_total/1024/1024} MB = {hw_bytes_total/1024/1024/1024} GB")
        print(f"需要内存大小= {self.total_data_bytes/1024/1024} MB = {self.total_data_bytes/1024/1024/1024} GB")
        assert self.total_data_bytes <= hw_bytes_total, f"数据量超过了硬件的内存大小: {self.total_data_bytes} Bytes > {hw_bytes_total} Bytes"

        # Number of vectors stored in each bank.
        self.num_vec_per_bank = math.ceil(self.data_num / self.hw_total_bank_for_mapping)

        # Number of vectors stored in each rank.
        self.num_vec_per_rank = self.num_vec_per_bank* SimConfig.bg * SimConfig.ba
        print(f"每个Bank放多少个向量: {self.num_vec_per_bank}, 每个Rank放多少个向量: {self.num_vec_per_rank}")

        # Number of vectors stored in each channel.
        self.vector_per_channel = self.num_vec_per_rank * SimConfig.ra

        # Vectors are distributed across devices in a rank, each device stores a slice.
        self.dim_per_device = int(self.data_dim / SimConfig.de)  # Number of dimensions handled by each device.
        vec_bytes_per_device = self.dim_per_device * (self.data_bits // 8)
        self.vec_columns_per_device =  math.ceil(vec_bytes_per_device*8/SimConfig.co_w)      # Number of bank columns per device needed to read one vector.
        print(f"每个Device分担的维度数: {self.dim_per_device}, 每个Device分担的数据字节数: {vec_bytes_per_device} Bytes, 每个Device分担的列数: {self.vec_columns_per_device}")

    # Map a vector ID to its rank and the row/column location in each device of that rank.
    # Data is interleaved across devices.
    def get_vector_location(self,vector_id):
        channel_id = vector_id // self.vector_per_channel
        vector_offset_in_channel = vector_id % self.vector_per_channel
        rank_id = vector_offset_in_channel // self.num_vec_per_rank
        vec_offset_in_rank_id = vector_offset_in_channel % self.num_vec_per_rank
        # In-rank mapping: which bank/row/column.
        bank_id = vec_offset_in_rank_id // self.num_vec_per_bank
        # if bank_id==16:
        #     print(f"Warning: Bank ID is 16, which is out of range. Check your vector ID: {vec_offset_in_rank_id}/{self.num_vec_per_bank}")
        vec_id_in_bank_offset = vec_offset_in_rank_id % self.num_vec_per_bank
        num_vec_per_row = int(SimConfig.co/self.vec_columns_per_device) # Number of vectors each bank row can store.
        row_id = vec_id_in_bank_offset//num_vec_per_row
        vec_id_in_col_offset = vec_id_in_bank_offset%num_vec_per_row
        col_id = vec_id_in_col_offset * self.vec_columns_per_device

        return channel_id, rank_id, bank_id, row_id, col_id

    def get_row_reuse_percentage(self,vec_idx_trace):

        last_rank_row_act={}
        row_reuse_count = 0

        for vec_idx_list in vec_idx_trace:
            for vec_idx in vec_idx_list:
                ch, ra, bank, row, col = self.get_vector_location(vec_idx)
                global_rank_id = ch * SimConfig.ra + ra
                row_in_rank_id = bank * SimConfig.ro + row
                if global_rank_id in last_rank_row_act:
                    if last_rank_row_act[global_rank_id] == row_in_rank_id:
                        row_reuse_count += 1
                    else:
                        last_rank_row_act[global_rank_id] = row_in_rank_id
                else:
                    last_rank_row_act[global_rank_id] = row_in_rank_id

        print(f"Row reuse count: {row_reuse_count / len(vec_idx_trace)*100}%")
        return row_reuse_count / len(vec_idx_trace)*100

    # vec_idx_trace records which vectors are accessed.
    # used_dims_trace records how many dimensions are used for those vectors.
    def gen_cmd_trace(self,vec_idx_trace,batch_size,used_dims_trace,success_num,changed,top,updated_top,debug_info=False,baseline=True,cache_size=2000):
        self.mem_acc = 0
        self.merge_cpu = 0
        self.load_cpu = 0
        self.fetch_nbr = 0
        self.cache_size=cache_size
        # Record the trace position for each query.
        query_idx_in_trace = []
        pointer = 0
        CPU_lat = 0

        def get_success_count_per_rank(success_entry, ch, ra):
            """Support both global-rank and per-channel rank histograms."""
            if np.isscalar(success_entry):
                return success_entry

            if isinstance(success_entry, np.ndarray):
                success_entry = success_entry.tolist()

            if not isinstance(success_entry, (list, tuple)):
                raise TypeError(
                    f"Unsupported success trace type: {type(success_entry)}"
                )

            global_rank_num = SimConfig.ch * SimConfig.ra
            if len(success_entry) == global_rank_num:
                return success_entry[ch * SimConfig.ra + ra]
            if len(success_entry) == SimConfig.ra:
                return success_entry[ra]

            raise ValueError(
                "Unexpected success trace length: "
                f"{len(success_entry)} (expected {global_rank_num} or {SimConfig.ra})"
            )

        hit_num = 0
        try_num = 0
        fetch_num = 0

        # When baseline=False, we frequently check whether used_top_node/used_updated_top
        # is contained in any cache_pack. The multiset (count table) reduces this check
        # from O(len(cache)) to O(1).
        if not baseline:
            self.cache_elem_cnt = {}
            for cache_pack in self.cache:
                for elem in cache_pack:
                    self.cache_elem_cnt[elem] = self.cache_elem_cnt.get(elem, 0) + 1

        for trace in vec_idx_trace:
            if trace=="LoadQuery":
                query_idx_in_trace.append(pointer)
            pointer += 1

        num_query = len(query_idx_in_trace)
        batch_num = math.ceil(num_query / batch_size)
        query_idx_in_trace.append(len(vec_idx_trace)) # Append one extra index because ranges are used below.

        # batch->hop
        total_hops = 0

        all_unindp_cmd_list = []
        for batch_id in range(batch_num):
            # Query traces in this batch.
            query_trace=[]
            # used_dims traces in this batch.
            query_used_dim=[]
            query_success_num=[]
            query_changed=[]
            query_top=[]
            query_updated_top=[]
            # Maximum BFS hop count in this batch (each hop is a list in the trace).
            max_query_bfs_len = 0
            # Number of queries in this batch.
            this_batch_size= batch_size if batch_id < batch_num - 1 else num_query - batch_id * batch_size
            for query_id in range(this_batch_size):
                # Extract trace data for this query.
                trace_begin_idx=query_idx_in_trace[batch_id * batch_size + query_id]
                trace_end_idx=query_idx_in_trace[batch_id * batch_size + query_id + 1]
                query_trace.append(vec_idx_trace[trace_begin_idx : trace_end_idx])
                query_used_dim.append(used_dims_trace[trace_begin_idx : trace_end_idx])
                query_success_num.append(success_num[trace_begin_idx : trace_end_idx])
                query_changed.append(changed[trace_begin_idx : trace_end_idx])
                query_top.append(top[trace_begin_idx : trace_end_idx])
                query_updated_top.append(updated_top[trace_begin_idx : trace_end_idx])
                # Update maximum BFS hop count.
                max_query_bfs_len = max(max_query_bfs_len, len(query_trace[query_id]))

            total_hops += max_query_bfs_len    
            
            # Generate execution traces hop by hop for each query.
            for hop in range(max_query_bfs_len):
                # Command list for each channel.
                each_ch_cmd_list = [[] for _ in range(SimConfig.ch)]
                # Number of distance computations on each rank per channel.
                rank_vs_dist_cal_num = [[0 for _ in range(SimConfig.ra)] for _ in range(SimConfig.ch)]
                used_top_node = [[] for _ in range(this_batch_size)]
                success_trace = [[] for _ in range(this_batch_size)]
                changed_trace = [[] for _ in range(this_batch_size)]
                used_updated_top = [[] for _ in range(this_batch_size)]
                # Process the current hop for each query.
                for query_id in range(this_batch_size):
                    # Skip if this query does not have this hop.
                    if(hop >= len(query_trace[query_id])):
                        continue
                    # Get the corresponding trace entry.
                    trace = query_trace[query_id][hop]
                    used_dim_trace = query_used_dim[query_id][hop]  # For LoadQuery, query_used_dim appears to append 0.
                    success_trace[query_id] = query_success_num[query_id][hop]
                    changed_trace[query_id] = query_changed[query_id][hop]
                    used_top_node[query_id] = query_top[query_id][hop]
                    used_updated_top[query_id] = query_updated_top[query_id][hop]

                    ########### Load Query or BFS trace ##########
                    if trace=="LoadQuery":
                        n_burst_each_rank = math.ceil(self.dim_per_device*32/SimConfig.co_w)
                        # Load query data to each rank.
                        for ch in range(SimConfig.ch):
                            for ra in range(SimConfig.ra):
                                self.load_cpu += n_burst_each_rank
                                for nburst in range(n_burst_each_rank):
                                    each_ch_cmd_list[ch].append((LEVEL.SYS, OPTYPE.host_write_rank_pu_reg, ch, ra, [True for _ in range(SimConfig.de)]))
                        
                                if query_id != 0:
                                    bank = 0
                                    row = 0
                                    col = 0
                                    used_col = self.vec_columns_per_device
                                    self.mem_acc += used_col
                                    each_ch_cmd_list[ch].append((LEVEL.RA, OPTYPE.pu, ch, ra, (SimConfig.de, [True for _ in range(SimConfig.de)]), (0, bank, row, col), (0, 0, 0, 0), used_col, False))

                    else:
                        for vec_idx, dim_used in zip(trace, used_dim_trace):
                            ch, ra, bank, row, col = self.get_vector_location(vec_idx)
                            used_col = math.ceil(dim_used * self.vec_columns_per_device / self.data_dim)
                            if self.do_dfloat:
                                used_col = math.ceil(dim_used / self.col_size[0])
                                for i in range(self.bits_type):
                                    if dim_used > self.dims_bound[i]:
                                        used_col = math.ceil((dim_used - self.dims_bound[i]) / self.col_size[i + 1]) + self.col_bound[i]
                                    else:
                                        break
                                used_col = math.ceil(used_col / 4)
                            self.mem_acc += used_col
                            each_ch_cmd_list[ch].append((LEVEL.RA, OPTYPE.pu, ch, ra, (SimConfig.de, [True for _ in range(SimConfig.de)]), (0, bank, row, col), (0, 0, 0, 0), used_col, False))

                            rank_vs_dist_cal_num[ch][ra] += 1

                ########### Read computation results back to CPU ##########
                if baseline:
                    for ch in range(SimConfig.ch):
                        for ra in range(SimConfig.ra):
                            data_bits = rank_vs_dist_cal_num[ch][ra]*32.0*2
                            n_burst_each_rank = math.ceil(data_bits/SimConfig.de/SimConfig.co_w)
                            self.merge_cpu += n_burst_each_rank
                            for t in range(n_burst_each_rank):
                                each_ch_cmd_list[ch].append(
                                    (LEVEL.SYS, OPTYPE.host_read_rank_pu_reg, ch, ra, [True for _ in range(SimConfig.de)])
                                )
                    CPU_lat += 240
                else:
                    for query_id in range(this_batch_size):
                        if(hop >= len(query_trace[query_id])):
                            continue
                        for ch in range(SimConfig.ch):
                            for ra in range(SimConfig.ra):
                                success_cnt = get_success_count_per_rank(
                                    success_trace[query_id], ch, ra
                                )
                                data_bits = success_cnt * 32.0 * 2
                                n_burst_each_rank = math.ceil(data_bits/SimConfig.de/SimConfig.co_w)
                                self.merge_cpu += n_burst_each_rank
                                for t in range(n_burst_each_rank):
                                    each_ch_cmd_list[ch].append(
                                        (LEVEL.SYS, OPTYPE.host_read_rank_pu_reg, ch, ra, [True for _ in range(SimConfig.de)])
                                    )

                ########### Format into simulator input ##########
                unindp_cmd_list = []
                for ch in range(SimConfig.ch):
                    if len(each_ch_cmd_list[ch]) != 0:
                        unindp_cmd_list.append((ch, [], each_ch_cmd_list[ch]))

                all_unindp_cmd_list.append(unindp_cmd_list)

                each_ch_cmd_list = [[] for _ in range(SimConfig.ch)]

                if baseline:
                    for ch in range(SimConfig.ch):
                        for ra in range(SimConfig.ra):
                            data_bits = 32 * 32.0
                            n_burst_each_rank = math.ceil(data_bits/4.0/SimConfig.co_w)
                            self.fetch_nbr += n_burst_each_rank
                            for t in range(n_burst_each_rank):
                                each_ch_cmd_list[ch].append(
                                    (LEVEL.SYS, OPTYPE.host_read_rank_pu_reg, ch, ra, [True for _ in range(SimConfig.de)])
                                )
                else:
                    for query_id in range(this_batch_size):
                        if(hop >= len(query_trace[query_id])):
                            continue
                        cache_elem_cnt = self.cache_elem_cnt
                        cached = cache_elem_cnt.get(used_top_node[query_id], 0) > 0
                        cache_threshold_active = len(self.cache) == self.cache_size
                        if changed_trace[query_id] and cache_threshold_active:
                            try_num += 1
                        if cached and cache_threshold_active:
                            hit_num += 1
                        self.pf_num[hop] += 1
                        if not changed_trace[query_id]:
                            self.pf_hit[hop] += 1
                        
                        if changed_trace[query_id] and not cached:
                            if cache_threshold_active:
                                fetch_num += 1
                            for ch in range(SimConfig.ch):
                                for ra in range(SimConfig.ra):
                                    each_ch_cmd_list[ch].append((LEVEL.RA, OPTYPE.pu, ch, ra, (SimConfig.de, [True for _ in range(SimConfig.de)]), (0, 1, 1, 1), (0, 0, 0, 0), 1, False))     
                                    data_bits = 32 / SimConfig.ch / SimConfig.ra * 32.0
                                    n_burst_each_rank = math.ceil(data_bits/SimConfig.de/SimConfig.co_w)
                                    self.fetch_nbr += n_burst_each_rank + 1
                                    each_ch_cmd_list[ch].append((LEVEL.RA, OPTYPE.pu, ch, ra, (SimConfig.de, [True for _ in range(SimConfig.de)]), (0, 1, 1, 1), (0, 0, 0, 0), n_burst_each_rank, False)) 
                            pack_to_add = self.cache_mapping[used_top_node[query_id]]
                            if len(self.cache) < self.cache_size:
                                self.cache.append(pack_to_add)
                            else:
                                old_pack = (
                                    self.cache.popleft()
                                    if hasattr(self.cache, "popleft")
                                    else self.cache.pop(0)
                                )
                                for elem in old_pack:
                                    new_cnt = cache_elem_cnt.get(elem, 0) - 1
                                    if new_cnt <= 0:
                                        cache_elem_cnt.pop(elem, None)
                                    else:
                                        cache_elem_cnt[elem] = new_cnt
                                self.cache.append(pack_to_add)
                            for elem in pack_to_add:
                                cache_elem_cnt[elem] = cache_elem_cnt.get(elem, 0) + 1
                        
                        cached = cache_elem_cnt.get(used_updated_top[query_id], 0) > 0
                        if not cached and used_updated_top[query_id] != 0:
                            pack_to_add = self.cache_mapping[used_updated_top[query_id]]
                            if len(self.cache) < self.cache_size:
                                self.cache.append(pack_to_add)
                            else:
                                old_pack = (
                                    self.cache.popleft()
                                    if hasattr(self.cache, "popleft")
                                    else self.cache.pop(0)
                                )
                                for elem in old_pack:
                                    new_cnt = cache_elem_cnt.get(elem, 0) - 1
                                    if new_cnt <= 0:
                                        cache_elem_cnt.pop(elem, None)
                                    else:
                                        cache_elem_cnt[elem] = new_cnt
                                self.cache.append(pack_to_add)
                            for elem in pack_to_add:
                                cache_elem_cnt[elem] = cache_elem_cnt.get(elem, 0) + 1
                                
                unindp_cmd_list = []
                for ch in range(SimConfig.ch):
                    if len(each_ch_cmd_list[ch]) != 0:
                        unindp_cmd_list.append((ch, [], each_ch_cmd_list[ch]))

                all_unindp_cmd_list.append(unindp_cmd_list)
            

        if try_num != 0:
            return all_unindp_cmd_list, CPU_lat, hit_num / try_num
        else:
            return all_unindp_cmd_list, CPU_lat, 0
