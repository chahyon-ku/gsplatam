

## Environment
```bash
mamba create -n jaxsplatam
mamba activate jaxsplatam
mamba install -c pytorch -c nvidia\
    python=3.12\
    cuda=11.8 cuda-cccl=11.8 cuda-nvcc=11.8 gxx_linux-64=11 cmake ninja\
    pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=11.8\
    jax jaxlib=*=cuda118*\
    tqdm opencv imageio matplotlib kornia natsort pyyaml wandb lpips torchmetrics\
    pytorch-msssim plyfile nvtx plotly ipykernel opencv rich
pip install --no-deps\
    open3d\
    git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110\
    git+https://github.com/chahyon-ku/diff-gaussian-rasterization-taming\
    git+https://github.com/nerfstudio-project/gsplat.git
pip install --no-deps -e\
    third_party/splatam\
    .
```

## Evaluate
```bash
bash third_party/splatam/SplaTAM/bash_scripts/download_replica.sh
bash third_party/splatam/SplaTAM/bash_scripts/download_replicav2.sh
bash third_party/splatam/SplaTAM/bash_scripts/download_tum.sh
python third_party/splatam/SplaTAM/scripts/splatam.py configs/replica/splatam_s_short.py
python scripts/gsplat_splatam.py configs/replica/splatam_s_short.py
```