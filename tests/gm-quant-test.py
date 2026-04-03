from datasets import load_dataset
from gptqmodel import AWQQuantizeConfig, GPTQModel

model_id = "meta-llama/Meta-Llama-3-8B"
quant_path = "Meta-Llama-3-8B-gptqmodel-4bit"

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = "Llama-2-7b-hf-gptqmodel-4bit"


calibration_dataset = load_dataset(
    "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train"
).select(range(1024))["text"]

quant_config = AWQQuantizeConfig(bits=4, group_size=64, sym=False)
# quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)
