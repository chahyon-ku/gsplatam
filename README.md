#

## Environment
```bash
mamba create -n jaxsplatam
mamba activate jaxsplatam
mamba install -c pytorch -c nvidia\
    python\
    cuda=11.8 cuda-cccl=11.8 cuda-nvcc=11.8 gxx_linux-64=11 cmake ninja\
    pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=11.8\
    tqdm opencv imageio matplotlib kornia natsort pyyaml wandb lpips torchmetrics\
    pytorch-msssim plyfile nvtx plotly ipykernel opencv rich
pip install --no-deps\
    open3d
git submodule update --init --recursive
pip install --no-deps -e\
    ./third_party/gsplat\
    ./third_party/splatam/SplaTAM/diff-gaussian-rasterization-w-depth.git\
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
nsys profile -o orig-replica-t python third_party/splatam/SplaTAM/scripts/splatam.py configs/replica/splatam_t.py &> orig-replica-t.log
nsys profile -o gsplat-replica-t python scripts/gsplat_splatam.py configs/replica/splatam_t.py &> gsplat-replica-t.log
nsys profile -o gsplat-tum-t python scripts/gsplat_splatam.py configs/tum/splatam_t.py &> gsplat-tum-t.log
nsys profile -o gsplat-tum python scripts/gsplat_splatam.py configs/tum/splatam.py &> gsplat-tum.log
nsys profile -o taming-replica-t python scripts/taming_splatam.py configs/replica/splatam_t.py &> taming-replica-t.log
```