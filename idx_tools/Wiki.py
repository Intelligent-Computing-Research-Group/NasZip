import yaml,os
import numpy as np
from idx_functions import load_fvecs,load_ivecs,read_fbin,read_ibin
from dataset_manage import dataset_cfg_path

class Wiki:
    def __init__(self):
        cfg_path=dataset_cfg_path
        with open(cfg_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self.path=dataset_config["Wiki"]["PATH"]

    def load_data(self):
        data_path = f"{self.path}/base.1M.fbin"
        return read_fbin(data_path),"fp32"

    def load_query(self):
        query_path=f"{self.path}/queries.fbin"
        return read_fbin(query_path),"fp32"

    def load_gt(self):
        gt_path=f"{self.path}/groundtruth.1M.neighbors.ibin"
        return read_ibin(gt_path)

if __name__ == "__main__":
    data=wiki1M()
    gt=data.load_gt()
    x=data.load_data()
    q=data.load_query()
    print(gt[0].shape)
    print(x[0].shape)
    print(q[0][0])
    