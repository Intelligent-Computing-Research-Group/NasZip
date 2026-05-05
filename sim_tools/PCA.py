from idx_tools.idx_functions import load_fvecs
import numpy as np
from Dfloat import Dfloat
from idx_tools.dataset_manage import df_cfg_path
import math
import yaml

class PCA:

    def __init__(self, dim, sampling_info, dist_type, do_dfloat, dataset_name):
        self.dim = dim
        self.sampling_info = sampling_info
        self.dist_type = dist_type
        self.L_path = sampling_info["L_path"]
        self.E_path = sampling_info["E_path"]
        
        # Load PCA data
        self.L = load_fvecs(self.L_path)
        self.E = load_fvecs(self.E_path)

        # 
        self.epsi = []
        self.lmds = []
        self.epsi.append(1.0e10)  # Epsilon for the 0-th dimension.
        for i in range(self.dim):
            self.lmds.append(self.L[0][i])
            self.epsi.append(self.E[0][i])
        
        # print(self.epsi）
        
        # Compute prefix sums of eigenvalues.
        self.cdf_lmds = []
        self.compute_cdf_lmd()

        print(dataset_name)
        if do_dfloat:
            with open(df_cfg_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.bits_type = data[dataset_name]['bits_type']
            self.bits = data[dataset_name]['bits']
            self.dims_bound = data[dataset_name]['dims']

    def compute_cdf_lmd(self):
        sum = 0
        self.cdf_lmds.append(sum)
        for i in range(self.dim):
            sum+=self.lmds[i]
            self.cdf_lmds.append(sum)

    # Compute the scaling ratio. In PCA, epsilon is not set manually;
    # it is generated from an acceptable probability and pre-stored.
    # cdf_lmds is the prefix sum of eigenvalues. cdf_lmds[D] represents
    # the expected squared distance in the first D dimensions, while
    # cdf_lmds[self.dim] represents the expected squared distance of the full PCA vector.
    def ratio(self, D):
        if self.dist_type == "l2":
            if D == self.dim:
                return 1.0
            return self.cdf_lmds[D] / self.cdf_lmds[self.dim] * (1.0 + self.epsi[D]) * (1.0 + self.epsi[D])
        else:
            if D == self.dim:
                return 0
            return (self.cdf_lmds[self.dim] - self.cdf_lmds[D]) / self.cdf_lmds[self.dim] * self.epsi[D] * 2.5
        
    
    # Compute incrementally to support early stopping.
    # If the condition is not met, return a negative estimated distance and used dimensions.
    # If the condition is met, return the exact distance and used dimensions.
    def l2_distance_sampling(self, vec1, vec2, lowerBound, do_dfloat=False):
        res = 0
        for i in range(int(self.dim / 4)):
            subv1 = vec1[i * 4 : (i + 1) * 4]
            subv2 = vec2[i * 4 : (i + 1) * 4]
            if do_dfloat:
                dfloat = Dfloat()
                subv1 = dfloat.mask_fp32_mantissa(subv1, 32 - self.bits[0])
                subv2 = dfloat.mask_fp32_mantissa(subv2, 32 - self.bits[0])
                for j in range(self.bits_type):
                    if i * 4 >= self.dims_bound[j]:
                        subv1 = dfloat.mask_fp32_mantissa(subv1, 32 - self.bits[j + 1])
                        subv2 = dfloat.mask_fp32_mantissa(subv2, 32 - self.bits[j + 1])
                    else:
                        break

            diff = np.array(subv1) - np.array(subv2)
            res += np.dot(diff, diff)
            if (i + 1) * 4 < self.dim: 
                if(res >= lowerBound * self.ratio((i + 1) * 4)):
                    return -res * self.cdf_lmds[self.dim] / self.cdf_lmds[(i + 1) * 4], (i + 1) * 4
        return res, self.dim
    
    def ip_distance_sampling(self, vec1, vec2, lowerBound, do_dfloat=False):
        res = 0
        norm = math.sqrt(np.dot(vec1, vec1) * np.dot(vec2, vec2))
        for i in range(int(self.dim / 5)):
            subv1 = vec1[i * 5 : (i + 1) * 5]
            subv2 = vec2[i * 5 : (i + 1) * 5]
            if do_dfloat:
                dfloat = Dfloat()
                subv1 = dfloat.mask_fp32_mantissa(subv1, 32 - self.bits[0])
                subv2 = dfloat.mask_fp32_mantissa(subv2, 32 - self.bits[0])
                for j in range(self.bits_type):
                    if i * 5 >= self.dims_bound[j]:
                        subv1 = dfloat.mask_fp32_mantissa(subv1, 32 - self.bits[j + 1])
                        subv2 = dfloat.mask_fp32_mantissa(subv2, 32 - self.bits[j + 1])
                    else:
                        break
            res += np.dot(subv1, subv2)
            
            if (i + 1) * 5 < self.dim:
                if - res - norm * self.ratio((i + 1) * 5) >= lowerBound:
                    return False, res, (i + 1) * 5 
        return True, -res, self.dim
