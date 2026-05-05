import yaml,os
import numpy as np
from idx_functions import load_fvecs,load_ivecs,read_fbin,read_ibin
from dataset_manage import dataset_cfg_path

class MS_MARCO:
    def __init__(self):
        cfg_path=dataset_cfg_path
        with open(cfg_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self.path=dataset_config["MS_MARCO"]["PATH"]

    def load_data(self):
        data_path = f"{self.path}/collection.fbin"
        return read_fbin(data_path),"fp32"

    def load_query(self):
        query_path=f"{self.path}/queries.eval.fbin"
        return read_fbin(query_path),"fp32"


    def load_gt(self):
        gt_path=f"{self.path}/gt.ibin"
        return read_ibin(gt_path)

if __name__ == "__main__":
    data=MS_MARCO()
    gt=data.load_gt()
    print(gt.shape)
    print(gt[0])
    print(data.load_data()[0].shape)
    