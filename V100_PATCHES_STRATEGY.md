# Stratégie de Patchs pour Support V100 (SM 7.0)

**Date** : 07/02/2026
**Version vLLM de base** : v0.9.2
**Objectif** : Support stable pour Tesla V100

## 🎯 Philosophie

Appliquer **uniquement** les patchs strictement nécessaires pour V100, en privilégiant :

- ✅ **Stabilité** plutôt que performance maximale
- ✅ **TORCH_SDPA** (backend natif PyTorch) plutôt que Triton
- ✅ **Modifications ciblées** plutôt que patchs intrusifs
- ❌ **ÉVITER** les modifications des kernels MLA/Attention Triton

## 📋 Patchs Catégorisés

### 1. Patchs CRITIQUES (Obligatoires)

#### 1.1 Variables de Compilation CUDA

```bash
# CRITIQUE : Utiliser CUDA 12.8 (dernière version avec support Volta)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Hints CMake
export CUDAToolkit_ROOT=/usr/local/cuda-12.8
export CMAKE_PREFIX_PATH=/usr/local/cuda-12.8

# Architecture cible
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=$(nproc)
export NVCC_THREADS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
```

**Raison** :

- **CUDA 13.0 ne supporte PAS SM 7.0** - Erreur `Unsupported gpu architecture 'compute_70'`
- **CUDA 12.8 est optimal** : Match parfait avec PyTorch 2.10.0+cu128, dernière version feature-complete pour Volta
- **Alternative** : CUDA 12.4 fonctionne aussi mais 12.8 réduit risques d'incompatibilités runtime

#### 1.2 Variables Runtime NCCL

```bash
export NCCL_CUMEM_ENABLE=0
export NCCL_PROTO=simple
```

**Raison** :

- `NCCL_CUMEM_ENABLE=0` : Désactive cuMemMap (instable sur Volta avec CUDA 13)
- `NCCL_PROTO=simple` : Garantit l'initialisation robuste des groupes de processus

#### 1.3 Backend Attention par Défaut

```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
```

**Raison** :

- FlashAttention v2 nécessite SM >= 8.0 (incompatible V100)
- TORCH_SDPA est stable et compatible SM 7.0

### 2. Patchs RECOMMANDÉS (Fortement conseillés)

#### 2.1 Dtype par Défaut

```python
# Dans client ou config
dtype = "half"  # Jamais "bfloat16" (nécessite SM 8.0)
```

**Raison** : V100 ne supporte pas nativement bfloat16

#### 2.2 Désactivation FlashInfer MoE

```bash
export VLLM_USE_FLASHINFER_MOE_FP16=0
```

**Raison** : FlashInfer est instable sur SM 7.0

### 3. Patchs OPTIONNELS (Seulement si erreurs)

#### 3.1 Triton num_stages pour Volta

**À appliquer UNIQUEMENT si** : Erreurs `OutOfResources` ou `illegal memory access`

**Fichiers concernés** :

- `vllm/model_executor/layers/fused_moe/fused_moe.py`
- `vllm/v1/attention/ops/triton_*.py` (si existants dans v0.9.2)

**Modification** :

```python
# Avant
num_stages = 2  # ou 4, ou 10

# Après (pour Volta uniquement)
device_capability = torch.cuda.get_device_capability(device)
if device_capability[0] == 7:  # Volta
    num_stages = 1
    num_warps = 4
else:
    num_stages = 2  # valeur originale
```

**Raison** : Volta a moins de mémoire partagée que Ampere/Hopper

### 4. Patchs INTERDITS (Ne jamais appliquer)

❌ **Force Triton MLA Backend** : Cause du garbage output sur V100
❌ **Modifications intrusives des kernels d'attention** : Déstabilise le moteur
❌ **Désactivation complète de backends** : Réduit la compatibilité

## 🧪 Ordre d'Application Recommandé

1. **Étape 1** : Appliquer les patchs CRITIQUES (#1.1, #1.2, #1.3)
2. **Étape 2** : Compiler vLLM avec `pip wheel`
3. **Étape 3** : Tester avec modèle simple (Qwen-0.5B) en mode Single-GPU
4. **Étape 4** : Si erreurs, appliquer les patchs RECOMMANDÉS (#2.1, #2.2)
5. **Étape 5** : Si erreurs Triton, appliquer patchs OPTIONNELS (#3.1)
6. **Étape 6** : Valider Multi-GPU (TP=2, TP=4)

## 📊 Validation de Succès

### Tests Minimaux

```bash
# Test 1 : Import
python3.12 -c "import vllm; print(vllm.__version__)"

# Test 2 : Single-GPU
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.5

# Test 3 : Multi-GPU (TP=2)
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --dtype half \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

### Critères de Validation

✅ Import vLLM sans erreur
✅ Démarrage moteur sans crash
✅ Génération texte **COHÉRENT** (pas de garbage)
✅ Communication NCCL stable en Multi-GPU

## 🚨 Debugging

Si le texte généré est corrompu (garbage) :

1. Vérifier que `VLLM_ATTENTION_BACKEND=TORCH_SDPA` est activé
2. Vérifier que `dtype="half"` (pas bfloat16)
3. Tester avec Transformers pur (validation matériel/poids)
4. **NE PAS** ajouter de patchs Triton MLA

Si crash au démarrage :

1. Vérifier `NCCL_CUMEM_ENABLE=0`
2. Vérifier compilation avec `TORCH_CUDA_ARCH_LIST="7.0"`
3. Vérifier logs : `/tmp/vllm_build.log`

## 📚 Références

- [TN-001 : Stratégie Stack Hybride V100](../../docs/TECH_NOTES/TN-001-v100-hybrid-stack.md)
- [Plan-310 : Journal Debugging vLLM V1](../../docs/PLANS/plan-310-Highbrain-Inference-Manager-HIM.md)
- [vLLM Issue #730 : BF16 V100](https://github.com/vllm-project/vllm/issues/730)
- [vLLM Issue #25456 : V100 CUDA Error](https://github.com/vllm-project/vllm/issues/25456)
