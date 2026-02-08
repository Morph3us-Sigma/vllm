# vLLM for Tesla V100 (SM 7.0)

**Fork** : https://github.com/Morph3us-Sigma/vllm
**Branch** : `v100-support`
**Base** : vLLM v0.9.2 (official release)
**Author** : Morph3us Sigma
**Date** : February 2026

---

## 🎯 Purpose

This fork provides a **configuration-only build** of vLLM v0.9.2 optimized for **Tesla V100 GPUs** (SM 7.0 / Volta architecture).

**Important** : This is **NOT a code fork** - we do NOT modify vLLM source code. We only provide build scripts and configuration for Volta compatibility.

---

## 🔑 Key Findings

### CUDA 13.0 Volta Deprecation

**CRITICAL** : CUDA 13.0 removed SM 7.0 (Volta) compilation support.

```bash
# CUDA 13.0 error
nvcc fatal: Unsupported gpu architecture 'compute_70'
```

### Solution : CUDA 12.8

**CUDA 12.8** is the optimal choice for V100:
- ✅ Last feature-complete CUDA version with Volta support
- ✅ Perfect match with PyTorch 2.10.0+cu128
- ✅ Latest libraries (cuBLAS 12.8.4, cuSPARSE 12.5.8, etc.)
- ✅ Maximum compatibility with vLLM runtime

---

## 📦 What's Included

### Files in This Fork

1. **`build_v100.sh`** : Automated build script
   - Sets CUDA 12.8 environment
   - Configures compilation flags for SM 7.0
   - Builds wheel with proper NCCL settings

2. **`V100_PATCHES_STRATEGY.md`** : Strategy documentation
   - Minimal patches approach
   - Environment variables explained
   - Troubleshooting guide

3. **`README_V100.md`** : This file

### What We DON'T Modify

- ❌ NO source code changes in vLLM
- ❌ NO custom kernels
- ❌ NO Triton patches
- ❌ NO MLA modifications

**Philosophy** : Configuration over modification.

---

## 🚀 Quick Start

### Prerequisites

- Tesla V100 GPU (SM 7.0)
- CUDA Toolkit 12.8 installed in `/usr/local/cuda-12.8`
- CUDA Driver R580+ (CUDA 13 capable for forward compatibility)
- Python 3.12
- PyTorch 2.10.0+cu128

### Build

```bash
# Clone this branch
git clone -b v100-support https://github.com/Morph3us-Sigma/vllm.git
cd vllm

# Run build script
./build_v100.sh
```

**Build time** : ~15-20 minutes (with 80 parallel jobs)

### Install

The script automatically installs the wheel. Or manually:

```bash
pip3.12 install dist/vllm-*.whl --force-reinstall --break-system-packages
```

---

## ✅ Validation

### Single-GPU Test

```bash
python3.12 -c "from vllm import LLM; \
  llm = LLM('Qwen/Qwen2.5-0.5B-Instruct', dtype='half', max_model_len=1024); \
  print(llm.generate(['Hello world'], max_tokens=20))"
```

**Expected** :
- Backend: XFormers (auto-selected)
- Engine: V0 (fallback for SM < 8.0)
- Coherent text output

### Multi-GPU Test (Tensor Parallel)

```bash
export NCCL_CUMEM_ENABLE=0
export NCCL_PROTO=simple

python3.12 -c "from vllm import LLM; \
  llm = LLM('mistralai/Mistral-7B-Instruct-v0.3', \
    dtype='half', tensor_parallel_size=2, max_model_len=2048, enforce_eager=True); \
  print(llm.generate(['Write a haiku'], max_tokens=50))"
```

**Expected** :
- NCCL 2.27.5 initialized (2 ranks)
- Model: ~6.8 GiB per GPU
- Coherent text output
- Performance: ~28 tokens/s

---

## 🏗️ Build Configuration

### Environment Variables (Critical)

```bash
# CUDA 12.8 (required for SM 7.0)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDAToolkit_ROOT=/usr/local/cuda-12.8
export CMAKE_PREFIX_PATH=/usr/local/cuda-12.8

# Volta architecture
export TORCH_CUDA_ARCH_LIST="7.0"

# Build parallelism
export MAX_JOBS=$(nproc)
export NVCC_THREADS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# NCCL for V100
export NCCL_CUMEM_ENABLE=0
export NCCL_PROTO=simple

# Attention backend (optional, auto-selected)
export VLLM_ATTENTION_BACKEND=XFORMERS

# FlashInfer disable (optional)
export VLLM_USE_FLASHINFER_MOE_FP16=0
```

---

## 📊 Validated Stack

This configuration has been tested and validated on:

**Hardware** :
- NVIDIA DGX-1 (8× Tesla V100 16GB)
- SM 7.0 (Volta)

**Software** :
- Ubuntu 24.04 LTS
- CUDA Toolkit 12.8.93
- CUDA Driver R580.126.09 (CUDA 13 capable)
- Python 3.12
- PyTorch 2.10.0+cu128
- xformers 0.0.34
- transformers 4.51.1

**Tests** :
- ✅ Single-GPU : Qwen-0.5B (coherent output, 55 tokens/s)
- ✅ Multi-GPU : Mistral-7B TP=2 (coherent output, 28 tokens/s)
- ✅ NCCL stable
- ✅ NO garbage output

---

## 🔧 Troubleshooting

### Problem: NVCC 13.0 Error

```
nvcc fatal: Unsupported gpu architecture 'compute_70'
```

**Solution** : Install CUDA 12.8 and update `CUDA_HOME`

### Problem: Garbage Output

**Symptoms** : Generated text contains random characters or `...Strings...`

**Solutions** :
1. Verify `dtype="half"` (NOT bfloat16 on V100)
2. Let vLLM auto-select backend (XFormers optimal for V100)
3. Rebuild with CUDA 12.8 (not 13.0)

### Problem: NCCL Multi-GPU Errors

**Symptoms** : `ncclInternalError` or `illegal memory access`

**Solutions** :
1. Set `NCCL_CUMEM_ENABLE=0`
2. Set `NCCL_PROTO=simple`
3. Verify both GPUs available: `nvidia-smi`

---

## 📚 Documentation

**Internal Docs** (HighBrain project) :
- [TN-001: V100 Hybrid Stack Strategy](../../docs/TECH_NOTES/TN-001-v100-hybrid-stack.md)
- [TN-003: CUDA 13 Volta Deprecation](../../docs/TECH_NOTES/TN-003-cuda-13-volta-deprecation.md)
- [Plan-310 Session 08/02/2026](../../docs/PLANS/plan-310-Session-08-02-2026.md)
- [Platinum Stack Installation](../../docs/STANDARDS/PLATINUM-STACK-INSTALLATION.md)

**vLLM Official** :
- https://github.com/vllm-project/vllm
- https://docs.vllm.ai/

---

## 🎓 Lessons Learned

1. **CUDA 13.0 is incompatible** : SM 7.0 support completely removed
2. **CUDA 12.8 is optimal** : Last Volta-compatible release
3. **No code patches needed** : vLLM v0.9.2 works perfectly with proper environment
4. **XFormers is best** : Auto-selected by vLLM, optimal for V100
5. **Forward compatibility works** : R580 driver (CUDA 13) runs CUDA 12.8 binaries

---

## 📋 Version History

### v0.9.2-v100-cuda12.8 (08/02/2026)

- ✅ CUDA 12.8 configuration
- ✅ Removed unnecessary code patches
- ✅ Configuration-only approach validated
- ✅ Single-GPU and Multi-GPU tests passed
- ✅ NO garbage output

### Previous Attempts (Archived)

- v0.9.2-v100-cuda12.4 : Worked but suboptimal (PyTorch mismatch)
- Earlier : Attempted code patches (triton_kernels, flash_mla) - unnecessary

---

## 🤝 Contributing

This is a personal fork for Tesla V100 support. If you have improvements:

1. Test thoroughly on V100 hardware
2. Ensure NO code modifications (config-only)
3. Document changes in `V100_PATCHES_STRATEGY.md`
4. Submit PR with test results

---

## 📄 License

Same as vLLM project (Apache 2.0)

---

## 🙏 Acknowledgments

- **vLLM Team** : For excellent inference engine
- **NVIDIA** : For V100 GPUs and CUDA toolkit
- **Claude Sonnet 4.5** : AI pair programming assistant

---

**Last updated** : February 8, 2026
**Maintainer** : Morph3us Sigma (morph3us.sigma@gmail.com)
**Status** : ✅ Production Ready
