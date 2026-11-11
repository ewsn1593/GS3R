# clip-vit-base-patch32 / 8 tasks / trunc
cuda=1
seeds=(40 41 42 43 44 45 46 47 48 49)

rank=256

for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        fabric.loggers.root_dir=outputs/gs3r/gs3r_8_Rank${rank} \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method=gs3r/GS3R \
        method.seed=${seeds[version]} \
        method.k=${rank} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done


