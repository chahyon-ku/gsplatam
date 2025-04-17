
backend=gsplat
size=tiny
data=replica_debug
gaussian_distribution=isotropic
nsys profile --force-overwrite true -o $backend-$size\
    python scripts/train.py\
        backend@_global_=$backend\
        data@_global_=$data\
        size@_global_=$size\
        gaussian_distribution=$gaussian_distribution

backend=orig
nsys profile --force-overwrite true -o $backend-$size\
    python scripts/train.py\
        backend@_global_=$backend\
        data@_global_=$data\
        size@_global_=$size\
        gaussian_distribution=$gaussian_distribution
