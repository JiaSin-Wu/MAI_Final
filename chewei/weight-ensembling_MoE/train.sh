docker run --rm --gpus all \
--shm-size=16g \
-v /data1/chliu:/data1/chliu \
-v /data1/jeffreytsai:/data1/jeffreytsai \
-v $(pwd):/weight-ensembling_MoE \
wemoe \
python -u scripts/clip_dictmoe_direct.py \
    model=ViT-L-14 \
    data_location=/data1/chliu/MAI2025_final/data \
    version=5 \
    tta=true \
    evaluate=false \
    num_devices=1