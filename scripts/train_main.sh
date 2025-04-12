mkdir -p logs
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
            size@_global_=$size
    done
    
    data=tum
    for size in tiny small base tum
    do
        echo "Training $backend on $data with model size $size"
        python scripts/train.py backend@_global_=$backend\
            data@_global_=$data\
            size@_global_=$size
    done
done