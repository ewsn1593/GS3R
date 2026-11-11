# flan-t5-base / glue tasks / trunc
cuda=0
seeds=(40 41 42 43 44 45 46 47 48 49)
k=64

for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
        fabric.loggers.root_dir=outputs/gs3r/gs3r_glue_Rank${k} \
        fabric.loggers.name=flan-t5-base \
        fabric.loggers.version=${version} \
        method=gs3r/GS3RFLAN \
        method.seed=${seeds[version]} \
        method.k=${k} \
        method.iter_num=0 \
        modelpool=Seq2SeqLMPool/flan-t5-base_glue \
        taskpool=flan-t5_glue_text_generation
done


