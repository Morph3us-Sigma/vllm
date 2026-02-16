#!/bin/bash
# Script de Compilation vLLM v0.9.2 pour Tesla V100 (SM 7.0)
# Date: 07/02/2026
# Auteur: Morph3us & AI.IDE
# Référence: README_V100.md

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  vLLM v0.9.2 Build pour Tesla V100 (SM 7.0)              ║${NC}"
echo -e "${BLUE}║  HighBrain Custom Build - Stratégie Patchs Minimaux      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# 1. Variables de Compilation CRITIQUES (Patch #1.1)
# ============================================================================

# CRITIQUE : CUDA 13.0 ne supporte PAS SM 7.0 (Volta)
# Forcer l'utilisation de CUDA 12.8 pour la compilation (match PyTorch cu128)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Hint CMake pour trouver libnvrtc dans CUDA 12.8
export CUDAToolkit_ROOT=/usr/local/cuda-12.8
export CMAKE_PREFIX_PATH=/usr/local/cuda-12.8

export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=$(nproc)
export NVCC_THREADS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

echo -e "${BLUE}[CONFIG] Variables de compilation V100 :${NC}"
echo -e "  - CUDA_HOME: ${YELLOW}${CUDA_HOME}${NC}"
echo -e "  - TORCH_CUDA_ARCH_LIST: ${YELLOW}${TORCH_CUDA_ARCH_LIST}${NC}"
echo -e "  - MAX_JOBS: ${YELLOW}${MAX_JOBS}${NC}"
echo -e "  - NVCC_THREADS: ${YELLOW}${NVCC_THREADS}${NC}"

# Vérification version NVCC
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
echo -e "${BLUE}[CHECK] Version NVCC: ${YELLOW}${NVCC_VERSION}${NC}"

if [[ "$NVCC_VERSION" == "13.0" ]]; then
    echo -e "${RED}[ERROR] NVCC 13.0 ne supporte PAS SM 7.0 (Volta) !${NC}"
    echo -e "${RED}[ERROR] Vérifiez que CUDA_HOME pointe vers CUDA 12.8${NC}"
    exit 1
fi

# ============================================================================
# 2. Variables Runtime NCCL (Patch #1.2)
# ============================================================================
export NCCL_CUMEM_ENABLE=0
export NCCL_PROTO=simple

echo -e "${BLUE}[CONFIG] Variables NCCL pour V100 :${NC}"
echo -e "  - NCCL_CUMEM_ENABLE: ${YELLOW}${NCCL_CUMEM_ENABLE}${NC}"
echo -e "  - NCCL_PROTO: ${YELLOW}${NCCL_PROTO}${NC}"

# ============================================================================
# 3. Backend Attention par Défaut (Patch #1.3)
# ============================================================================
# XFORMERS est optimal pour V100 (SM 7.0)
export VLLM_ATTENTION_BACKEND=XFORMERS

echo -e "${BLUE}[CONFIG] Backend Attention :${NC}"
echo -e "  - VLLM_ATTENTION_BACKEND: ${YELLOW}${VLLM_ATTENTION_BACKEND}${NC}"

# ============================================================================
# 4. Désactivation FlashInfer MoE (Patch #2.2)
# ============================================================================
export VLLM_USE_FLASHINFER_MOE_FP16=0

echo -e "${BLUE}[CONFIG] Désactivations FlashInfer :${NC}"
echo -e "  - VLLM_USE_FLASHINFER_MOE_FP16: ${YELLOW}${VLLM_USE_FLASHINFER_MOE_FP16}${NC}"

# ============================================================================
# 5. Nettoyage des builds précédents
# ============================================================================
echo -e "${BLUE}[CLEAN] Nettoyage des builds précédents...${NC}"
rm -rf build/ dist/ *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# 6. Installation des dépendances de build
# ============================================================================
echo -e "${BLUE}[DEPS] Installation des dépendances de build...${NC}"

# Vérifier si requirements-build.txt existe
if [ -f "requirements-build.txt" ]; then
    sudo pip3.12 install -r requirements-build.txt \
        --break-system-packages \
        --ignore-installed packaging || true
else
    echo -e "${YELLOW}[WARN] requirements-build.txt non trouvé, installation manuelle...${NC}"
fi

# Dépendances essentielles
sudo pip3.12 install \
    packaging ninja wheel setuptools_scm python-dotenv \
    --break-system-packages \
    --ignore-installed packaging

# ============================================================================
# 7. Compilation du Wheel
# ============================================================================
echo -e "${BLUE}[BUILD] Compilation du Wheel vLLM v0.9.2 pour V100...${NC}"
echo -e "${BLUE}[BUILD] Timestamp début: $(date)${NC}"
echo -e "${YELLOW}[INFO] Cette opération peut prendre 15-30 minutes...${NC}"

LOG_FILE="/tmp/vllm_v100_build_$(date +%Y%m%d_%H%M%S).log"

python3.12 -m pip wheel . \
    --no-deps \
    --no-build-isolation \
    -w dist \
    -v 2>&1 | tee "$LOG_FILE"

# ============================================================================
# 8. Vérification et Installation du Wheel
# ============================================================================
echo -e "${BLUE}[INSTALL] Recherche du Wheel compilé...${NC}"
WHEEL_PATH=$(find dist -name "vllm*.whl" | head -n 1)

if [ -z "$WHEEL_PATH" ]; then
    echo -e "${RED}[ERROR] Wheel non trouvé !${NC}"
    echo -e "${RED}[ERROR] Consultez les logs : ${LOG_FILE}${NC}"
    exit 1
fi

echo -e "${GREEN}[SUCCESS] Wheel trouvé: ${WHEEL_PATH}${NC}"
echo -e "${BLUE}[INSTALL] Installation du Wheel...${NC}"

sudo pip3.12 install "$WHEEL_PATH" \
    --force-reinstall \
    --break-system-packages \
    --ignore-installed packaging

# ============================================================================
# 9. Validation de l'installation
# ============================================================================
echo -e "${BLUE}[VERIFY] Vérification de l'installation...${NC}"

if python3.12 -c "import vllm; print(f'✅ vLLM version: {vllm.__version__}')"; then
    echo -e "${GREEN}[SUCCESS] vLLM installé avec succès !${NC}"
else
    echo -e "${RED}[ERROR] Import vLLM échoué !${NC}"
    exit 1
fi

# ============================================================================
# 10. Résumé Final
# ============================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ BUILD TERMINÉ AVEC SUCCÈS                             ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Version: vLLM v0.9.2 Custom V100                         ║${NC}"
echo -e "${GREEN}║  Architecture: SM 7.0 (Tesla V100)                        ║${NC}"
echo -e "${GREEN}║  Backend Attention: XFORMERS                              ║${NC}"
echo -e "${GREEN}║  NCCL: CUMEM_ENABLE=0, PROTO=simple                       ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  📝 Logs: ${LOG_FILE}${NC}"
echo -e "${GREEN}║  📦 Wheel: ${WHEEL_PATH}${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "${BLUE}[INFO] Timestamp fin: $(date)${NC}"
echo -e "${YELLOW}[NEXT] Pour tester : vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half${NC}"
