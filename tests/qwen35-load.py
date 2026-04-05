import torch
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    Qwen3_5ForConditionalGeneration,
)


def load_with_auto_model():
    model_id = "Qwen/Qwen3.5-9B"

    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return model


def load_with_std_transformers():
    model_id = "Qwen/Qwen3.5-9B"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return model


def load_with_qwen3_type():
    model_id = "Qwen/Qwen3.5-9B"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return model


def main():
    model = load_with_auto_model()
    print(model.config)
    # model = load_with_std_transformers()
    # print(model.config)
    # model = load_with_qwen3_type()
    # print(model.config)


if __name__ == "__main__":
    main()
