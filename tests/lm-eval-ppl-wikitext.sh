# Save results and model outputs
lm-eval run \
    --model hf \
    --model_args pretrained=Qwen/Qwen3.5-9B \
    --tasks wikitext \
    --output_path ./results/ \
    --log_samples \
    --batch_size auto \
    --max_batch_size 2

