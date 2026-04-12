# Save results and model outputs
lm-eval run \
    --model hf \
    --model_args pretrained=Qwen/Qwen3.5-9B \
    --tasks gpqa_diamond_zeroshot \
    --output_path ./results/ \
    --log_samples
