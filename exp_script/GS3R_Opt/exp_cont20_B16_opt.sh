# clip-vit-base-patch16
# eight tasks, ablation study on alpha
# without random seed, for each alpha, run 10 times
cuda=2
seeds=(40 41 42 43 44 45 46 47 48 49)

for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        fabric.loggers.root_dir=outputs/gs3r/gs3r_20_opt \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method=gs3r/GS3R \
        method.seed=${seeds[version]} \
        method.iter_num=1000 \
        method.save_on_every_step=true \
        method.evaluate_on_every_step=true \
        method.opt_only_last=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


