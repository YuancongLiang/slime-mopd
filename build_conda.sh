#!/bin/bash

set -ex

export SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PATH="/root/.local/bin:${PATH}"

# create conda
if [ ! -x /root/.local/bin/micromamba ]; then
  yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
fi
export PS1=tmp
mkdir -p /root/.cargo/
touch /root/.cargo/env
source ~/.bashrc

# The micromamba installer writes `nodefaults` into ~/.condarc as a channel
# entry, which newer micromamba versions try to fetch as a real anaconda.org
# repo (it isn't — it's a meta-tag) and time out on. Strip it.
if [ -f ~/.condarc ]; then
  sed -i '/^\s*-\s*nodefaults\s*$/d' ~/.condarc
fi

if micromamba env list | awk '{print $1}' | grep -qx slime; then
  micromamba install -n slime python=3.12 pip -c conda-forge -y
else
  micromamba create -n slime python=3.12 pip -c conda-forge -y
fi
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"

export GITHUB_PROXY="${GITHUB_PROXY:-https://v4.gh-proxy.com/}"

# Keep these in sync with docker/Dockerfile:
#   - SGLANG_IMAGE_TAG (ARG)            -> SGLANG_VERSION below
#   - MEGATRON_COMMIT (ARG)             -> MEGATRON_COMMIT below
#   - PATCH_VERSION (ARG, default "latest") -> PATCH_VERSION below
#   - TMS_COMMIT (ARG)                   -> TMS_COMMIT below
#   - FLASH_QLA_COMMIT (ARG)             -> FLASH_QLA_COMMIT below
export SGLANG_VERSION="v0.5.15.post1"
export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="v0.5.15.post1"
export TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"
export FLASH_QLA_COMMIT="821fd9d37ede18fdc2a4e707fefe3770bfc32e58"

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n slime \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge \
  -y
micromamba install -n slime -c conda-forge cudnn -y
# sglang's editable install builds a Rust extension and generates its gRPC
# bindings, so the conda env needs rustc, cargo, and protoc.
micromamba install -n slime -c conda-forge rust protobuf libprotobuf -y
export PROTOC="$CONDA_PREFIX/bin/protoc"

# install sglang. The Dockerfile starts FROM slimerl/sglang:v0.5.15.post1-cu129
# which already has sglang installed with cu129-built native kernels; we have
# to install it ourselves here. Three follow-up steps clean up the cu13 spill:
#   1. restore cuda-python 12.9 after SGLang's >=13 dependency upgrades it;
#   2. force-reinstall torch / sglang-kernel / sgl-deep-gemm to their +cu129
#      wheels (pypi defaults are cu13);
#   3. uninstall the cu13 nvidia-* runtime libs sglang dragged in, then
#      reinstall the cu12 equivalents to repair the `site-packages/nvidia/*`
#      shared dirs (pip uninstall stomps libs co-owned across cu12/cu13).
if [ ! -d "$BASE_DIR/sglang" ]; then
  cd $BASE_DIR
  git clone "${GITHUB_PROXY}https://github.com/sgl-project/sglang.git"
fi
cd $BASE_DIR/sglang
git checkout ${SGLANG_COMMIT}
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall "cuda-python==12.9"
pip install --force-reinstall --no-deps "cuda-toolkit==12.9.1"
pip install --force-reinstall --no-deps \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url https://docs.sglang.ai/whl/cu129/
pip uninstall -y \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cuda-nvcc \
  nvidia-cuda-tileiras \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-cccl-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple


pip install cmake ninja

# The SGLang dependency set includes FA4, while the validated TE 2.16 CP stack
# uses FA2 2.7.4 through 2.8.3. This wheel targets Python 3.12, PyTorch 2.11,
# CUDA 12, and the CXX11 ABI used by the conda environment. Upstream does not
# publish that combination, so pin the community build linked from upstream
# issue #2425 by its SHA256 digest.
pip uninstall -y flash-attn-4 flash_attn_4 || true
pip install --no-deps \
  "${GITHUB_PROXY}https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"

# Its declared torch dependency pins older CUDA runtime wheels. All runtime
# dependencies are already provided by SGLang, so keep the cu129 stack intact.
pip install --no-deps flash-linear-attention==0.4.2
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
pip install "git+${GITHUB_PROXY}https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT}" \
  --no-deps --no-build-isolation
# tilelang (matches Dockerfile)
pip install --no-deps tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

pip install onnx onnxscript
pip install --no-deps nvdlfw-inspect
pip install --no-deps --no-build-isolation \
  "transformer_engine==2.16.1" \
  "transformer_engine_cu12==2.16.1" \
  "transformer_engine_torch==2.16.1"

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  "git+${GITHUB_PROXY}https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
# --no-build-isolation: TMS's setup.py needs to find nvcc + headers + the
# installed torch to build its cu${TMS_CUDA_MAJOR} native hook; pip's default
# PEP 517 build venv hides them, so the wheel comes out python-only (~46KB)
# and sglang trips `Only hook_mode=preload supports pauseable CUDA Graph`
# because the preload .so was never compiled in.
pip install -v "git+${GITHUB_PROXY}https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT}" \
  --no-cache-dir --force-reinstall --no-deps --no-build-isolation
# SGLang already provides torch and its runtime dependencies. ModelOpt's torch
# extra would otherwise re-resolve the cu129 stack against PyPI's CUDA pins.
pip install --no-deps "nvidia-modelopt>=0.37.0" --no-build-isolation
pip install --no-deps --force-reinstall \
  "${GITHUB_PROXY}https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl"
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# megatron
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone "${GITHUB_PROXY}https://github.com/NVIDIA/Megatron-LM.git" --recursive
fi
# pre-install Megatron's build deps explicitly since we use --no-build-isolation
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
# --no-build-isolation: setup.py builds a C++ extension (megatron.core.datasets.helpers_cpp)
# that subprocess-shells `python3 -m pybind11`; without isolation pip uses the
# current env's python which already has pybind11 installed. Otherwise the ext
# is marked optional and silently skipped, which breaks GPT dataset loading.
cd $BASE_DIR/Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
  pip install -e . --no-deps --no-build-isolation

# install slime and apply patches

cd $SLIME_DIR
# Install slime's pure-python runtime deps first (wandb, ray, accelerate,
# transformers, etc.) from its requirements.txt, then install slime itself
# with --no-deps so pip doesn't re-resolve and stomp our pinned native libs
# (torch+cu129, sglang-kernel+cu129, ...). The Dockerfile does the same thing
# in two RUN layers (line ~71 + line ~124).
pip install --no-deps -r requirements.txt
# Resolve the lightweight dependencies of packages that are not part of
# SGLang's dependency set without asking pip to re-resolve torch/CUDA.
pip install \
  textual connectrpc dockerfile-parse h2 "protobuf-py>=0.1.1,<0.2" \
  pyqwest wcmatch "pyjwt[crypto]>=2.10.1" \
  "sse-starlette>=3.0" platformdirs sentry-sdk absl-py markdown \
  tensorboard-data-server werkzeug "llvmlite>=0.49.0dev0,<0.50" msgpack \
  "openai==2.6.1" "openai-agents==0.4.2" "mcp[cli]==1.29.1" \
  "websockets>=15,<17" "setuptools>=80,<82" socksio \
  aiohttp-cors colorful "py-spy>=0.4.0" opencensus \
  "opentelemetry-sdk>=1.30.0" opentelemetry-exporter-prometheus \
  opentelemetry-proto prometheus-client smart-open \
  "virtualenv!=20.21.1,>=20.0.24"
pip install --no-deps "fla-core==0.4.2"
pip install -e . --no-deps

# int4_qat kernel (matches Dockerfile)
cd $SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat
pip install . --no-build-isolation

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy==1.26.4" "scipy==1.17.1"
pip install "ml-dtypes<0.6" "tifffile<2026"
# kernels 0.15.x trips a ValueError("Either a revision or a version must be
# specified") on `transformers.integrations.hub_kernels` import; pin to <0.15
# so `import sglang` works at runtime.
pip install "kernels<0.15.0"

# apply patches in the same order as Dockerfile
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
if [ ! -d "$patch_dir" ]; then
  echo "Patch directory does not exist: $patch_dir" >&2
  exit 1
fi

cd $BASE_DIR/sglang
for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch sglang-deterministic.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    continue
  fi
  if git apply --check "$patch_path"; then
    git apply "$patch_path"
  elif git apply --reverse --check "$patch_path"; then
    echo "$patch_name already applied, skipping"
  else
    echo "$patch_name does not apply cleanly" >&2
    exit 1
  fi
done
cd $BASE_DIR/Megatron-LM
for patch_name in megatron.patch megatron-sglang-aligned.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    if [ "$patch_name" = "megatron.patch" ]; then
      echo "Megatron patch does not exist: $patch_path" >&2
      exit 1
    fi
    continue
  fi

  if git apply --reverse --check "$patch_path"; then
    echo "$patch_name already applied, skipping"
  else
    git update-index --refresh || true
    if ! git apply "$patch_path" --3way; then
      echo "$patch_name does not apply cleanly" >&2
      exit 1
    fi
    if git grep -n '^<<<<<<< ' -- .; then
      echo "$patch_name failed to apply cleanly. Please resolve conflicts." >&2
      exit 1
    fi
  fi
done

python - <<'PY'
import sglang
import torch
import torchaudio
import torchvision

assert torch.__version__ == "2.11.0+cu129"
assert torchaudio.__version__ == "2.11.0+cu129"
assert torchvision.__version__ == "0.26.0+cu129"
assert hasattr(torch.ops.torchvision, "nms")
PY
