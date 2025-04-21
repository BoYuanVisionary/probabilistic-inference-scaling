NUM_PARTICLES=(8)

# adaptive
for P in ${NUM_PARTICLES[@]}; do
        python ./scripts/pg.py \
        --total-timesteps 1 \
        --n-particles $P \
        --dataset-start 0 \
        --dataset-end 100 \
        --prm-path Qwen/Qwen2.5-Math-PRM-7B \
        --softmax-temp 1 \
        --seed 96 \
        --model-path meta-llama/Llama-3.2-1B-Instruct \
        --output-dir ./output/subset100_threshold95_adaptive_p$P/ \
	--resample-inactive \
        --resampling-threshold 0.95 \
        --dataset-path /ssdscratch/byuan48/particle_filtering/probabilistic-inference-scaling/datasets/math500.jsonl
done

# baseline
for P in ${NUM_PARTICLES[@]}; do
        python ./scripts/pg.py \
        --total-timesteps 1 \
        --n-particles $P \
        --dataset-start 0 \
        --dataset-end 100 \
        --prm-path Qwen/Qwen2.5-Math-PRM-7B \
        --softmax-temp 1 \
        --seed 96 \
        --model-path meta-llama/Llama-3.2-1B-Instruct \
        --output-dir ./output/subset100_baseline_p$P/ \
        --dataset-path /ssdscratch/byuan48/particle_filtering/probabilistic-inference-scaling/datasets/math500.jsonl
done

