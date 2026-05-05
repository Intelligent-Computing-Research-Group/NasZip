import yaml
import numpy as np
from idx_functions import load_fvecs,load_ivecs
from dataset_manage import dataset_cfg_path

class SIFT:
    def __init__(self):
        cfg_path=dataset_cfg_path
        with open(cfg_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self.path=dataset_config["SIFT"]["PATH"]
        self.distance_type = "L2"

    def load_data(self,dtype="fp"):
        data_path=f"{self.path}/sift_base.fvecs"
        if dtype=="fp":
            return load_fvecs(data_path).astype(np.float32, copy=False) / 255.0,"fp32"
        elif dtype=="uint":
            return load_fvecs(data_path),"uint8"

    def load_query(self,dtype="fp"):
        query_path=f"{self.path}/sift_query.fvecs"

        if dtype=="fp":
            return load_fvecs(query_path,copy=True).astype(np.float32, copy=False) / 255.0,"fp32"
        elif dtype=="uint":
            return load_fvecs(query_path,copy=True),"uint8"

    def load_gt(self):
        gt_path=f"{self.path}/sift_groundtruth.ivecs"
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