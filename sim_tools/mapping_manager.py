


def get_mapping(mapping_name,ddr_cfg_path, data_num, dim, do_dfloat, dataset_name, cache_mapping=None):
    if mapping_name == "vec_in_rank_interleave":
        from in_rank_interleave_mapping import VecInRankInterleaveMapping
        return VecInRankInterleaveMapping(ddr_cfg_path, data_num, dim, do_dfloat, dataset_name, cache_mapping=cache_mapping)
    else:
        raise ValueError(f"Unsupported mapping type: {mapping_name}")