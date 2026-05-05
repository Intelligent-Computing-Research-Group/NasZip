cd preprocess_idx

# Usage: bash build_index.sh [GPU_ID]
GPU_ID="${1:-0}"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building SIFT index with PCA sampling and Dx generation${NC}"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d SIFT \
-gd 16 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s 

echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building SIFT index with PCA sampling"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d SIFT \
-gd 32 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s

echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building SIFT index with PCA sampling"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d SIFT \
-gd 64 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s


echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building GIST index with PCA sampling and Dx generation${NC}"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d GIST \
-gd 64 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s 


echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building GloVe index with PCA sampling and Dx generation${NC}"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d GloVe \
-gd 16 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s 

echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building Wiki index with PCA sampling and Dx generation${NC}"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d Wiki \
-gd 32 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s 

echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building MS_MARCO index with PCA sampling and Dx generation${NC}"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d MS_MARCO \
-gd 32 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s 

echo -e "${CYAN}================================================${NC}"
echo -e "${BOLD}${GREEN}Building BigANN100M index with PCA sampling and Dx generation${NC}"
echo -e "${CYAN}================================================${NC}"
python index_build_via_cagra.py \
-d BigANN100M \
-gd 32 \
-ef 200 \
-gpu_id "${GPU_ID}" \
-s