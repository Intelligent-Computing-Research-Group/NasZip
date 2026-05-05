import yaml
import numpy as np
from idx_functions import load_fvecs,load_ivecs
from dataset_manage import dataset_cfg_path

class GIST:
    def __init__(self):
        cfg_path=dataset_cfg_path
        with open(cfg_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self.path=dataset_config["GIST"]["PATH"]

    def load_data(self):
        data_path=f"{self.path}/gist_base.fvecs"
        return load_fvecs(data_path),"fp32"

    def load_query(self):
        query_path=f"{self.path}/gist_query.fvecs"
        return load_fvecs(query_path,copy=True),"fp32"

    def load_gt(self):
        gt_path=f"{self.path}/gist_groundtruth.ivecs"
        return load_ivecs(gt_path,copy=True)

# if __name__ == "__main__":
#     gist=gist1M()
#     data=gist.load_data()
#     print(data[0])
#     print(data.dtype)
#     query=load_query()
#     print(query.shape)
#     print(query.dtype)
#     gt=load_gt()
#     print(gt.shape)
#     print(gt[0])