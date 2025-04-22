NUM_PARTICLES=(2)
THRESHOLD=(0.5 0.999)
SEEDS=(0 1 2)

# adaptive
for P in ${NUM_PARTICLES[@]}; do
        for SEED in ${SEEDS[@]}; do
                OUTPUT_DIR="./output/ap$P/seed$SEED/"
                mkdir -p ${OUTPUT_DIR}
                python ./scripts/pg_adaptive.py \
                --total-timesteps 1 \
                --n-particles $P \
                --dataset-start 0 \
                --dataset-end 100 \
                --prm-path Qwen/Qwen2.5-Math-PRM-7B \
                --softmax-temp 1 \
                --seed $SEED \
                --model-path meta-llama/Llama-3.2-3B-Instruct \
                --output-dir ${OUTPUT_DIR} \
                --resample-inactive \
                --dataset-path /ssdscratch/byuan48/particle_filtering/probabilistic-inference-scaling/datasets/math500.jsonl
                &> ${OUTPUT_DIR}/run.log
        done
done


# # Baseline
# for P in ${NUM_PARTICLES[@]}; do
#         for SEED in ${SEEDS[@]}; do
#         OUTPUT_DIR="./output/p$P/seed$SEED/"
#         mkdir -p ${OUTPUT_DIR}
#         python ./scripts/pg.py \
#         --total-timesteps 1 \
#         --n-particles $P \
#         --dataset-start 0 \
#         --dataset-end 500 \
#         --prm-path Qwen/Qwen2.5-Math-PRM-7B \
#         --softmax-temp 1 \
#         --seed $SEED \
#         --model-path meta-llama/Llama-3.2-1B-Instruct \
#         --output-dir ${OUTPUT_DIR} \
#         --dataset-path /ssdscratch/byuan48/particle_filtering/probabilistic-inference-scaling/datasets/math500.jsonl \
#         &> ${OUTPUT_DIR}/run.log
#         done
# done

