## Todo

* Generate current logs/profiles (done)
* Compare not packed vs. packed (done): packed is faster when there are more gaussians contrary to documentation
* Hydra configs (done)
* Single-step multi-view mapping
* sgd / visible adam / persistent adam
    * separate out tracking and mapping  optimizers
* Isometric CUDA kernels (dropped)
* Taming-cached CUDA kernels (dropped)
* Entropy-based keyframe selection based on Joey Wilson's paper

## Environment
```bash
git submodule update --init --recursive
mamba create -n gsplatam -c pytorch -c nvidia\
    python\
    cuda=12.1 cuda-version=12.1 cuda-cccl=12.1 cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1\
    gxx_linux-64=11 cmake ninja\
    pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.1\
    tqdm opencv imageio matplotlib kornia natsort pyyaml wandb lpips torchmetrics\
    pytorch-msssim plyfile nvtx plotly ipykernel opencv rich hydra-core
mamba activate gsplatam
pip install --no-deps\
    open3d\
    gsplat==1.4
pip install --no-deps -e\
    ./third_party/splatam\
    ./third_party/splatam/SplaTAM/diff-gaussian-rasterization-w-depth.git
    ./third_party/fused-ssim\
    .
```

## Download Data
```bash
bash third_party/splatam/SplaTAM/bash_scripts/download_replica.sh
bash third_party/splatam/SplaTAM/bash_scripts/download_tum.sh
```

## Evaluate
```bash
# gsplat-splatam-tiny (4/6 iterations) on replica
python scripts/gsplat_splatam.py configs/old/replica/splatam_t.py
# gsplat-splatam on replica
python scripts/gsplat_splatam.py configs/old/replica/splatam.py
# splatam on replica
python third_party/splatam/SplaTAM/scripts/splatam.py configs/old/replica/splatam.py
```

## Benchmark
```bash
# link error -lcuda
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"

bash scripts/train_all.sh

backend=gsplat
size=tiny
data=replica
nsys profile --wait primary -o $backend\_$size-$data --force-overwrite true\
    python scripts/train.py\
    backend=$backend\
    data@_global_=$data\
    size@_global_=$size\
    2>&1 | tee $backend\_$size-$data\.log
```