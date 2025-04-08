mkdir -p logs-orig
python third_party/splatam/SplaTAM/scripts/splatam.py\
    third_party/splatam/SplaTAM/configs/replica/splatam_s.py\
    | tee logs-orig/orig_small-replica.log
python third_party/splatam/SplaTAM/scripts/splatam.py\
    third_party/splatam/SplaTAM/configs/replica/splatam.py\
    | tee logs-orig/orig_base-replica.log