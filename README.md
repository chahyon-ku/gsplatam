## Todo

* Generate current logs/profiles (done)
* Compare not packed vs. packed (done): packed is faster when there are many gaussians?
* Single-step multi-view mapping
* Isometric CUDA kernels
* Taming-cached CUDA kernels

## Environment
```bash
mamba create -n jaxsplatam -c pytorch -c nvidia\
    python\
    cuda=12.1 cuda-version=12.1 cuda-cccl=12.1 cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1\
    gxx_linux-64=11 cmake ninja\
    pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.1\
    tqdm opencv imageio matplotlib kornia natsort pyyaml wandb lpips torchmetrics\
    pytorch-msssim plyfile nvtx plotly ipykernel opencv rich
mamba activate jaxsplatam
pip install --no-deps\
    open3d\
    git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110\
    git+https://github.com/nerfstudio-project/gsplat.git\
    git+https://github.com/chahyon-ku/diff-gaussian-rasterization-taming.git\
    gsplat
pip install --no-deps -e\
    ./third_party/splatam\
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
python scripts/gsplat_splatam.py configs/replica/splatam_t.py
# gsplat-splatam on replica
python scripts/gsplat_splatam.py configs/replica/splatam.py
# splatam on replica
python third_party/splatam/SplaTAM/scripts/splatam.py configs/replica/splatam.py
```

## Benchmark
```bash
# link error -lcuda
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"

python third_party/splatam/SplaTAM/scripts/splatam.py configs/replica/splatam.py &> orig-replica.log
python scripts/gsplat_splatam.py configs/replica/splatam.py &> gsplat-replica.log
nsys profile -o orig-replica-t python third_party/splatam/SplaTAM/scripts/splatam.py configs/replica/splatam_t.py &> orig-replica-t.log
nsys profile -o gsplat-replica-t python scripts/gsplat_splatam.py configs/replica/splatam_t.py &> gsplat-replica-t.log
nsys profile -o orig-tum-t python third_party/splatam/SplaTAM/scripts/splatam.py  configs/tum/splatam_t.py &> orig-tum-t.log
nsys profile -o gsplat-tum-t python scripts/gsplat_splatam.py configs/tum/splatam_t.py &> gsplat-tum-t.log
nsys profile -o gsplat-tum python scripts/gsplat_splatam.py configs/tum/splatam.py &> gsplat-tum.log

nsys profile -o taming-replica-t python scripts/taming_splatam.py configs/replica/splatam_t.py &> taming-replica-t.log
nsys profile -o taming-tum-t python scripts/taming_splatam.py configs/tum/splatam_t.py &> taming-tum-t.log

nsys profile -o gsplat_packed-replica-t python scripts/gsplat_splatam.py configs/replica/splatam_t.py &> gsplat_packed-replica-t.log
nsys profile -o gsplat_packed-tum-t python scripts/gsplat_splatam.py configs/tum/splatam_t.py &> gsplat_packed-tum-t.log
nsys profile -o gsplat_packed-tum-s python scripts/gsplat_splatam.py configs/tum/splatam_s.py &> gsplat_packed-tum-s.log
python scripts/gsplat_splatam.py configs/tum/splatam.py &> gsplat_packed-tum.log
```