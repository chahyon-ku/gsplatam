mkdir -p logs
python third_party/splatam/SplaTAM/scripts/splatam.py\
    third_party/splatam/SplaTAM/configs/replica/splatam_s.py\
    | tee logs/replica-orig_small.log
python third_party/splatam/SplaTAM/scripts/splatam.py\
    third_party/splatam/SplaTAM/configs/replica/splatam.py\
    | tee logs/replica-orig_base.log
python third_party/splatam/SplaTAM/scripts/splatam.py\
    third_party/splatam/SplaTAM/configs/tum/splatam.py\
    | tee logs/tum-orig_tum.log