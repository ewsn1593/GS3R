# flan-t5-base
# glue tasks
cuda=5
seeds=(40 41 42 43 44 45 46 47 48 49)

for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        fabric.loggers.root_dir=outputs/gs3r/gs3r_glue_opt \
        fabric.loggers.name=flan-t5-base \
        fabric.loggers.version=${version} \
        method=gs3r/GS3RFLAN \
        method.seed=${seeds[version]} \
        method.save_on_every_step=true \
        method.evaluate_on_every_step=true \
        method.opt_only_last=false \
        modelpool=Seq2SeqLMPool/flan-t5-base_glue \
        taskpool=flan-t5_glue_text_generation
done


