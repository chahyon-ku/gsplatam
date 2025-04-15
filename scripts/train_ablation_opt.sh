mkdir -p logs
# python scripts/train.py backend=orig | tee logs/replica-orig.log
python scripts/train.py backend@_global_=orig_parallel_dataloading