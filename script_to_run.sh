NUM_PARTICLES=(2)
HF_TOKEN={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}


for P in ${NUM_PARTICLES[@]}; do
        python ./scripts/pg.py \
        --total-timesteps 1 \
        --n-particles $P \
        --dataset-start 0 \
        --dataset-end 500 \
        --prm-path Qwen/Qwen2.5-Math-PRM-7B \
        --softmax-temp 1 \
        --seed 96 \
        --model-path meta-llama/Llama-3.2-1B-Instruct \
        --output-dir ./output/p$P/ \
	--resample-inactive \
        --dataset-path /ssdscratch/byuan48/particle_filtering/probabilistic-inference-scaling/datasets/math500.jsonl
done

#YOU ONLY HAVE TO CHANGE THE LOCATION OF PG.PY AND THE OUTPUT DIR.
