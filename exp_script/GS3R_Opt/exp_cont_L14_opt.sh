# clip-vit-large-patch14
# eight tasks, ablation study on alpha
# without random seed, for each alpha, run 10 times
cuda=2
seeds=(40 41 42 43 44 45 46 47 48 49)

for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        fabric.loggers.root_dir=outputs/gs3r/gs3r_8_opt \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method=gs3r/GS3R \
        method.seed=${seeds[version]} \
        method.save_on_every_step=true \
        method.evaluate_on_every_step=true \
        method.opt_only_last=false \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14
done


