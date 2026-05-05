import numpy as np
import torch,time,os
from contextlib import contextmanager
import struct,copy
try:
    import cupy as cp  # optional: only needed for some GPU/CuPy paths
except ModuleNotFoundError:
    cp = None

# numba is optional; some environments may fail to import it (e.g. numpy/ABI mismatch).
# When numba is unavailable, fall back to no-op jit and plain Python iteration.
try:
    from numba import jit, prange
except Exception:  # pragma: no cover
    def jit(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    prange = range

def load_fvecs(path,copy=False):
    x = np.memmap(path, dtype='int32', mode='r')
    d = x[0]
    data_view=x.reshape(-1,d+1)[:, 1:].view('float32')
    return data_view.copy() if copy else data_view

def load_ivecs(path,copy=False):
    x = np.fromfile(path, dtype='int32')
    d = x[0]
    data_view=x.reshape(-1, d + 1)[:, 1:]
    return data_view.copy() if copy else data_view

def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)

def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
        Args:
            :param filename (str): path to *.ibin file
            :param start_idx (int): start reading vectors from this index
            :param chunk_size (int): number of vectors to read.
                                     If None, read all vectors
        Returns:
            Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def load_bvecs(path,num_vec=None,copy=False):
    header = np.memmap(path, dtype='uint8', mode='r', shape=(4,), offset=0)
    d = header.view('int32')[0]
    vec_stride = d + 4 

    if num_vec is None:
        total_bytes = os.path.getsize(path)
    else:
        total_bytes = int(num_vec) * int(vec_stride)
    print(total_bytes)
    x = np.memmap(path,
                  dtype='uint8',
                  mode='r',
                  shape=(total_bytes,),
                  offset=0)

    arr = x.reshape(-1, vec_stride)[:, 4:]

    return arr.copy() if copy else arr

def cal_recall(query_res,gt,k=10):
    num_queries,_=query_res.shape
    assert num_queries==gt.shape[0]

    num_hit=0
    for i in range(num_queries):
        a=query_res[i][:k]
        b=gt[i][:k]
        num_hit+=len(set(a).intersection(set(b)))

    return num_hit/(num_queries*k)

def cupy_load(data, dtype, load_with_shard):
    import cupy as cp
    if load_with_shard:
        n, d = data.shape
        vecs = cp.empty((n, d), dtype=dtype)
        step  = 1_000_000                        
        for s in range(0, n, step):
            vecs[s:s+step] = cp.asarray(
                data[s:s+step], dtype=dtype)
    else:
        vecs = cp.asarray(data, dtype=dtype)

    return vecs



def merge_top_k(best_labels,best_distances, new_labels, new_distances, k=10,device=torch.device("cuda")):
    """
    Merge two sets of top-k results.
    """
    combined_labels = torch.cat((best_labels, new_labels),dim=1)
    combined_distances = torch.cat((best_distances, new_distances),dim=1)

    # Get the indices of the top-k elements
    # top_k_indices = np.argpartition(combined_distances, k)[:k]
    _,top_k_idx=torch.topk(combined_distances, k, dim=1,largest=False, sorted=False)

    # Sort the top-k elements
    merged_distances = torch.gather(combined_distances, 1, top_k_idx)
    merged_labels    = torch.gather(combined_labels,    1, top_k_idx)

    return merged_labels, merged_distances

@contextmanager
def cal_QPS(query_count):
    """
    Context manager to calculate QPS (Queries Per Second).
    Usage:
        with cal_QPS() as qps:
            # Your code here
            pass
        print(f"QPS: {qps}")
    """
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    qps = query_count / (end_time - start_time) if (end_time - start_time) > 0 else float('inf')
    print(f"QPS: {qps:.2f}")

@contextmanager
def get_duration():
    """
    Context manager to calculate QPS (Queries Per Second).
    Usage:
        with cal_QPS() as qps:
            # Your code here
            pass
        print(f"QPS: {qps}")
    """
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"Duration: {end_time - start_time:.2f} seconds")

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

def save_fvecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

# Generate PCA projection matrix.
# x is the transposed database matrix.
def PCA_cpu(x):
    x = np.array(copy.deepcopy(x), dtype=np.float32, copy=True)
    x -= x.mean(axis=1, keepdims=True)
    cov = x @ x.T
    lmd, w = np.linalg.eig(cov)
    return np.array(w, dtype=np.float32), np.array(lmd, dtype=np.float32)[np.newaxis, :]


# Generate PCA projection matrix.
# x is the transposed database matrix.
def PCA_gpu(x):
    x_gpu = torch.tensor(x).float().cuda()
    print("Dataset is transfered to GPU.")
    
    # Subtract the mean from x.
    x_gpu -= torch.mean(x_gpu, dim=1, keepdim=True)
    
    # Compute eigenvalues (lmd) and eigenvectors (w) of x * x.T.
    print("Doing MM on GPU...")
    mat_gpu = torch.matmul(x_gpu, x_gpu.T)
    mat_cpu = mat_gpu.cpu()  # Move results back to CPU.
    print("Doing PCA on GPU...")
    lmd, w = np.linalg.eig(mat_cpu)
    lmd = np.expand_dims(lmd, axis=0)

    del x_gpu
    torch.cuda.empty_cache()  # Clear GPU memory cache.

    return w,lmd
