import pandas as pd
from hqq.utils.optimizer import find_optimal_configs

from lm_quant_toolkit.eval.common import get_mxq_quant_meta_data_file


def dump_mxq_objectives(model_ids, bit_budgets, csv_fp="mxq-objectives.csv"):
    dikt = []
    for model_id in model_ids:
        short_id = model_id.split("/")[1]
        _, fp = get_mxq_quant_meta_data_file(model_id)
        for bit_budget in bit_budgets:
            _, objective = find_optimal_configs(fp, bit_budget, time_limit=200)
            dikt.append(
                {
                    "model": short_id,
                    "bpp": bit_budget,
                    "fnorm": objective,
                }
            )

    df = pd.DataFrame(dikt)
    df.to_csv(csv_fp, index=False)


def dump_mxq_configs(model_ids, bit_budgets, csv_fp, weight_algo, factor):
    dikt = []
    for model_id in model_ids:
        short_id = model_id.split("/")[1]
        for bit_budget in bit_budgets:
            try:
                _, fp = get_mxq_quant_meta_data_file(model_id)
                kwargs = {"weight_algo": weight_algo, "factor": factor}
                configs, objective = find_optimal_configs(
                    fp,
                    bit_budget,
                    time_limit=200,
                    **kwargs,
                )
                for k, v in configs.items():
                    comps = k.split(".", 1)
                    layer, module = comps[0], comps[1]
                    dikt.append(
                        {
                            "model": short_id,
                            "module": module,
                            "layer": layer,
                            "memmb": 0,  # to work with the plot program
                            "bit_budget": bit_budget,
                            "b1": v[0],
                            "g1": v[1],
                            "b2": v[2],
                            "g2": v[3],
                        }
                    )
            except ValueError:
                print(f"Warning: {bit_budget:.2f} unsolvable for model {model_id}")
    df = pd.DataFrame(dikt)
    df.to_csv(csv_fp, index=False)


if __name__ == "__main__":
    bit_budgets = [4.51, 4.25, 4.13]
    dump_mxq_configs(["meta-llama/Llama-2-7b-hf"], bit_budgets)
