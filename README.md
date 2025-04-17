<p align="center">

  <h1 align="center">gsplatam: Real-time Splat, Track, and Map with gsplat</h1>
  <h3 align="center">
    <strong>Chahyon Ku</strong>
    ,
    <strong>David Wang</strong>
    ,
    <strong>Ruihan Chen</strong>
    ,
    <strong>Yicheng Zou</strong>
  </h3>
  <h3 align="center"><a href="./assets/gsplatam-paper.pdf">Paper</a> | <a href="./assets/gsplatam-poster.pdf">Poster</a> | <a href="./assets/gsplatam-slides.pdf">Slides</a> | <a href="https://www.youtube.com/watch?v=T2NLEBzQ5w8">Video</a></h3>
  <div align="center"></div>
</p>

1. This project replaces the [gaussian splatting backbone](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/tree/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110) of [SplaTAM](https://github.com/spla-tam/SplaTAM) with [gsplat](https://github.com/nerfstudio-project/gsplat), with speed comparisons and 2DGS experiments.
2. This project provides [viser](https://github.com/nerfstudio-project/viser) visualization from offline datasets (replica and TUM), RGBD streams from iPhone - [Record3D app](https://record3d.app/), and iPhone - [NerfCapture app](https://github.com/jc211/NeRFCapture).

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


https://github.com/user-attachments/assets/605c50e5-f4f3-41bf-b574-8721e5e31ba5


https://github.com/user-attachments/assets/7f043d7a-064f-4835-ab37-f9012324474c



### iPhone - Record3D
```bash
python scripts/demo_record3d.py
```


https://github.com/user-attachments/assets/df651639-2a2e-4bbc-9a3d-ea028d871674


### iPhone - NerfCapture
```bash
python scripts/demo_nerfcapture.py
```

## Evaluate
```bash
bash scripts/train_orig.sh
bash scripts/train_ours.sh
```

