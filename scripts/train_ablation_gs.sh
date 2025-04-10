mkdir -p logs
python scripts/train.py backend@_global_=gsplat | tee logs/replica-gsplat.log
python scripts/train.py backend@_global_=gsplat gaussian_distribution=anisotropic | tee logs/replica-gsplat_aniso.log
python scripts/train.py backend@_global_=gsplat_2dgs | tee logs/replica-gsplat_2dgs.log
python scripts/train.py backend@_global_=gsplat_2dgs gaussian_distribution=anisotropic | tee logs/replica-gsplat_2dgs_aniso.log