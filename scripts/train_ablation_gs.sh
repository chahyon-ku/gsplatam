mkdir -p logs
# python scripts/train.py backend@_global_=gsplat gaussian_distribution=isotropic
# python scripts/train.py backend@_global_=gsplat gaussian_distribution=anisotropic
python scripts/train.py backend@_global_=gsplat_2dgs gaussian_distribution=isotropic
python scripts/train.py backend@_global_=gsplat_2dgs gaussian_distribution=anisotropic