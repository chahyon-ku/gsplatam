mkdir -p logs
for backend in gsplat orig
do
    data=tum
    for size in tiny small base tum # tiny small small2 base tum
    do
        echo "Training $backend on $data with model size $size"
        # nsys profile --force-overwrite true -o $backend\_$size-$data\
        python scripts/train.py backend=$backend\
            data@_global_=$data\
            size@_global_=$size\
            2>&1 | tee logs/$backend\_$size-$data\.log
    done
done