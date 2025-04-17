# gsplatam: Real-time Splat, Track, and Map with gsplat

This project replaces the gaussian splatting backbone of [SplaTAM](https://github.com/spla-tam/SplaTAM) with [gsplat](https://github.com/nerfstudio-project/gsplat)
and provide [viser](https://github.com/nerfstudio-project/viser) visualization with RGBD streams from an iPhone using either the [Record3D app](https://record3d.app/) or the [NerfCapture app](https://github.com/jc211/NeRFCapture).

## Environment Setup 
```bash
git submodule update --init --recursive
mamba create -n gsplatam -c pytorch -c nvidia\
    python\
    cuda=12.1 cuda-version=12.1 cuda-cccl=12.1 cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1\
    gxx_linux-64=11 cmake ninja\
    pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.1\
    tqdm opencv imageio matplotlib kornia natsort pyyaml wandb lpips torchmetrics\
    pytorch-msssim plyfile nvtx plotly ipykernel opencv rich hydra-core\
    dash scikit-learn addict pandas\
    websockets msgspec tyro jaxtyping
mamba activate gsplatam
pip install --no-deps\
    open3d\
    viser==0.02\
    nerfview
pip install --no-deps -e\
    ./third_party/splatam\
    ./third_party/splatam/SplaTAM/diff-gaussian-rasterization-w-depth.git\
    ./third_party/fused-ssim\
    ./third_party/gsplat\
    .
# link error -lcuda
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
```

## Download Data
```bash
bash third_party/splatam/SplaTAM/bash_scripts/download_replica.sh
bash third_party/splatam/SplaTAM/bash_scripts/download_tum.sh
```

## Visualization

### Offline Dataset
```bash
python scripts/train.py size@_global_=tiny data@_global_=replica viewer=True
python scripts/train.py size@_global_=base data@_global_=tum viewer=True
```

### iPhone - Record3D
```bash
python scripts/demo_record3d.py
```

### iPhone - NerfCapture
```bash
python scripts/iphone_demo_live.py
```

## Evaluate
```bash
bash scripts/train_orig.sh
bash scripts/train_ours.sh
```