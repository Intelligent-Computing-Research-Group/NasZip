import os,math
import yaml

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_cfg_path= f"{current_dir}/dataset_cfg.yaml"
df_cfg_path = f"{current_dir}/Dfloat.yaml"
# dataset_cfg_path= "dataset_cfg.yaml" # For testing in local environment

def gen_idx_name(dataset_name, index_type, type="CPU",graph_degree=None,ef_construction=None):
    with open(dataset_cfg_path, "r") as f:
        idx_path = yaml.safe_load(f)["IDX_CACHE"]["PATH"]

    if graph_degree is not None and ef_construction is not None:
        path= f"{idx_path}/{dataset_name}/{dataset_name}_{type}_{index_type}_gd{graph_degree}_ef{ef_construction}.bin"
    else:
        path= f"{idx_path}/{dataset_name}/{dataset_name}_{type}_{index_type}.bin"
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


# Shard format {"id": int, "len": int}
def get_data(dataset_name,shard=None,db_data=False,query=False,gt=False):
    if dataset_name == "SIFT":
        from SIFT import SIFT
        dataset = SIFT()
    elif dataset_name == "GIST":
        from GIST import GIST
        dataset = GIST()
    elif dataset_name == "GloVe":
        from GloVe import GloVe
        dataset = GloVe()
    elif dataset_name == "BigANN100M":
        from BigANN100M import BigANN100M
        dataset = BigANN100M()
    elif dataset_name == "Wiki":
        from Wiki import Wiki
        dataset = Wiki()
    elif dataset_name == "MS_MARCO":
        from MS_MARCO import MS_MARCO
        dataset = MS_MARCO()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are 'sift1M', 'gist1M', 'glove', 'wiki', 'msmacro'.")
    
    if db_data:
        return dataset.load_data() if shard is None else dataset.load_data(shard_id=shard["id"], shard_len=shard["len"])
    elif query:
        return dataset.load_query()
    elif gt:
        return dataset.load_gt()
    else:
        raise ValueError("You must specify at least one of db_data, query, or gt as True.")
    
def get_shard_info(dataset_name, shard_num_floor=True):
    with open(dataset_cfg_path, "r") as f:
        dataset_config = yaml.safe_load(f)[dataset_name]
        if dataset_config.get("SHARD") and dataset_config["SHARD"]==True:
            dataset_size = dataset_config["DATASET_SIZE"]
            shard_num_ceil = math.ceil(dataset_size/dataset_config["SHARD_SIZE"])
            shard_num_floor = math.floor(dataset_size/dataset_config["SHARD_SIZE"])
            shard_num = shard_num_floor if shard_num_floor else shard_num_ceil
            return (dataset_config["SHARD_SIZE"],shard_num)
        else:
            return None
        

def get_dataset_info(dataset_name):
    with open(dataset_cfg_path, "r") as f:
        dataset_config = yaml.safe_load(f)[dataset_name]
        return dataset_config["DATASET_SIZE"], dataset_config["DIM"]


def get_dist_type(dataset_name):
    with open(dataset_cfg_path, "r") as f:
        dataset_config = yaml.safe_load(f)[dataset_name]
        return dataset_config["DISTANCE_TYPE"]
    
def get_dataset_idx_location(dataset_name):
    with open(dataset_cfg_path, "r") as f:
        idx_path = yaml.safe_load(f)["IDX_CACHE"]["PATH"]

    path= f"{idx_path}/{dataset_name}/"
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return path