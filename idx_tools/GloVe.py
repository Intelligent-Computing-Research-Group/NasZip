import yaml,os
import numpy as np
from idx_functions import load_fvecs,load_ivecs
from dataset_manage import dataset_cfg_path

class GloVe:
    def __init__(self):
        cfg_path=dataset_cfg_path
        with open(cfg_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self.path=dataset_config["GloVe"]["PATH"]
        self.distance_type = "IP"
    
    def load_data(self):
        data_path=f"{self.path}/glove.twitter.27B.100d.fvecs"
        return load_fvecs(data_path),"fp32"
    
    def load_query(self):
        query_path=f"{self.path}/glove.twitter.27B.100d.query.fvecs"
        return load_fvecs(query_path,copy=True),"fp32"

    def load_gt(self):
        gt_path=f"{self.path}/glove.twitter.27B.100d.gt.ivecs"
        return load_ivecs(gt_path,copy=True)

if __name__ == "__main__":
    data=GloVe()
    print(data.load_gt())
    print(data.load_query())
    print(data.load_data())