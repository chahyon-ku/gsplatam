mkdir -p logs
# python scripts/train.py backend=orig 2>&1 | tee logs/replica-orig.log
python scripts/train.py backend@_global_=orig_parallel_dataloading 2>&1 | tee logs/replica-orig_parallel_dataloading.log
python scripts/train.py backend@_global_=gsplat 2>&1 | tee logs/replica-gsplat.log