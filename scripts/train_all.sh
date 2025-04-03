
for backend in orig gsplat
do
    for data in tum # tum replica
    do
        for size in base #tiny small small2 base tum
        do
            echo "Training $backend on $data with model size $size"
            nsys profile -o $backend\_$data\_$size python scripts/train.py backend=$backend\
                data@_global_=$data\
                size@_global_=$size\
                2>&1 | tee $backend\_$data\_$size\.log
        done
    done
done