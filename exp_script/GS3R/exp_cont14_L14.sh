# clip-vit-large-patch14 / 14 tasks
cuda=7
seeds=(40 41 42 43 44 45 46 47 48 49)

for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        fabric.loggers.root_dir=outputs/gs3r/gs3r_14 \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method=gs3r/GS3R \
        method.seed=${seeds[version]} \
        method.iter_num=0 \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done



