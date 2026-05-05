cd preprocess_idx

python get_varience.py --dataset_name SIFT  --use_gpu false
python get_varience.py --dataset_name GIST  --use_gpu false
python get_varience.py --dataset_name GloVe  --use_gpu false
python get_varience.py --dataset_name Wiki  --use_gpu false