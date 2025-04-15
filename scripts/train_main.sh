mkdir -p logs
for gaussian_distribution in anisotropic #isotropic anisotropic
do
    for backend in gsplat
    do
        data=replica
        for size in tiny small base
        do
            echo "Training $backend on $data with model size $size"
            # nsys profile --force-overwrite true -o $backend\_$size-$data\
            python scripts/train.py\
                backend@_global_=$backend\
                data@_global_=$data\
                size@_global_=$size\
                gaussian_distribution=$gaussian_distribution
        done
        
        data=tum
        for size in tiny small base tum
        do
            echo "Training $backend on $data with model size $size"
            python scripts/train.py backend@_global_=$backend\
                data@_global_=$data\
                size@_global_=$size\
                gaussian_distribution=$gaussian_distribution
        done
    done
done