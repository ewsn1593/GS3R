# clip-vit-base-patch16 / 8 tasks
cuda=3
seeds=(40 41 42 43 44 45 46 47 48 49)

for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        fabric.loggers.root_dir=outputs/gs3r/gs3r_8\
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method=gs3r/GS3R \
        method.seed=${seeds[version]} \
        method.iter_num=0 \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16
done


