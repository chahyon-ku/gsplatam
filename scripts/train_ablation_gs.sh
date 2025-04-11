mkdir -p logs
for size in tiny # tiny small
do
    for backend in gsplat_2dgs gsplat # gsplat gsplat_2dgs
    do
        for gaussian_distribution in anisotropic isotropic
        do
            nsys profile python scripts/train.py\
                backend@_global_=$backend\
                size@_global_=$size\
                gaussian_distribution=$gaussian_distribution\
                2>&1 | tee logs/replica-$backend-$gaussian_distribution-$size.log
        done
    done
done