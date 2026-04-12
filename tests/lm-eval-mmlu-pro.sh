# Save results and model outputs
lm-eval run \
    --model hf \
    --model_args pretrained=Qwen/Qwen3.5-9B \
    --tasks mmlu_pro \
    --output_path ./results/ \
    --seed 42 \
    --limit 1000 \
    --batch_size auto \
    --log_samples
