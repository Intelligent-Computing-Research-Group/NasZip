import yaml
import numpy as np
from idx_functions import load_fvecs,load_ivecs,load_bvecs
from dataset_manage import dataset_cfg_path

class BigANN100M:
    def __init__(self):
        cfg_path=dataset_cfg_path
        with open(cfg_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self.path=dataset_config["BigANN100M"]["PATH"]
        self.distance_type = "L2"

    def load_data(self,dtype="fp"):
        data_path=f"{self.path}/bigann_base_100M.bvecs"
        data = load_bvecs(data_path).astype(np.float32, copy=False) 
        if dtype=="fp":
            return data / 255.0,"fp32"
        elif dtype=="uint":
            return data,"uint8"

    def load_query(self,dtype="fp"):
        query_path=f"{self.path}/bigann_query.bvecs"
        query_data = load_bvecs(query_path,copy=True).astype(np.uint8, copy=False) 
    
        if dtype=="fp":
            return query_data / 255.0,"fp32"
        elif dtype=="uint":
            return query_data,"uint8"

    def load_gt(self):
        gt_path=f"{self.path}/gt_idx_100M.ivecs"
        return load_ivecs(gt_path,copy=True)


# if __name__ == "__main__":
#     data=load_data()
#     print(data[0])
#     print(data.dtype)
#     query=load_query()
#     print(query.shape)
#     print(query.dtype)
#     gt=load_gt()
#     print(gt.shape)
#     print(gt[0])