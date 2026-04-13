import os

from hqq.utils.optimizer import find_optimal_configs

from lm_quant_toolkit.utils.hub import LLAMA_MODELS, VIT_OPENCLIP_MODELS


def get_mxq_quant_meta_data_file(model_id):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    fp = os.path.join(data_dir, "fnorm-.csv")
    return os.path.exists(fp), os.path.abspath(fp)


def calc_bits(b1, g1, b2=8, g2=128):
    return b1 + 2 * b2 / g1 + 32 / g1 / g2


def plan_eval_bit_budgets(
    model_arch="ViT",
    points=5,
    step=1,
    bases=[4.51],
    include_base=False,
):
    for base in bases:
        ideals, solvables = get_eval_plan(model_arch, base, points, step, include_base)
        print("*" * 72)
        print(f"base: {base}")
        for t in zip(ideals, solvables):
            print(f"ideal: {t[0]:.2f}, solvable: {t[1]:.2f}")
        print("*" * 72)


def get_eval_plan(model_arch, base, points, step, include_base):
    ideals = []
    solvables = []
    start = 0 if include_base else 1
    for point in range(start, points + 1):
        tentative = round(base + point * step, 2)
        ideals.append(tentative)
        ret = try_solvable(model_arch, tentative, step)
        if ret is not None:
            solvables.append(ret)
        else:
            solvables.append(0.0)
    return ideals, solvables


def try_solvable(model_arch, bit_budget, step):
    if model_arch == "ViT":
        model_ids = VIT_OPENCLIP_MODELS.keys()
    else:
        model_ids = [
            key for key in LLAMA_MODELS if LLAMA_MODELS[key].get("experiment", False)
        ]

    dikt = {}
    feasible_budget = round(bit_budget, 2)
    for model_id in model_ids:
        attempts = 1
        _, fp = get_mxq_quant_meta_data_file(model_id)
        while True:
            try:
                # find_optimal_configs(fp, feasible_budget, time_limit=200)
                find_optimal_configs(
                    fp,
                    feasible_budget,
                    time_limit=200,
                    weight_algo="sensi-milp",
                )
                dikt[model_id] = feasible_budget
                break
            except ValueError:
                print(f"Warning: {feasible_budget:.2f} unsolvable for model {model_id}")
                if attempts > 3:
                    return None
                feasible_budget += -0.01 if step < 0 else 0.01
            attempts += 1
    if len(set(dikt.values())) > 1:
        for model_id, budget in dikt.items():
            if abs(budget - feasible_budget) > 0.01:
                try:
                    fp = get_mxq_quant_meta_data_file(model_id)
                    find_optimal_configs(fp, feasible_budget, time_limit=200)
                except ValueError:
                    print(
                        f"Warning: {feasible_budget:.2f} unsolvable for model {model_id}"
                    )
                    return None
    return feasible_budget


def debug_milp_solvable(model_id, bit_budget):
    _, fp = get_mxq_quant_meta_data_file(model_id)
    configs = find_optimal_configs(fp, bit_budget, time_limit=200)
    print(configs)


def plan_432_bits():
    bases = [4.51, 4.25, 4.13]
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=0.02, bases=bases, include_base=True
    )
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=-0.02, bases=bases, include_base=True
    )
    bases = [3.51, 3.25, 3.13, 2.51, 2.25, 2.13]
    plan_eval_bit_budgets(
        model_arch="llm", points=3, step=0.02, bases=bases, include_base=True
    )
    plan_eval_bit_budgets(
        model_arch="llm", points=3, step=-0.02, bases=bases, include_base=True
    )


def plan_567_bits():
    best_bit_budget = calc_bits(8, 32, 8, 128)
    save_objs = [10, 20, 30, 40]
    bases = [best_bit_budget * (100 - obj) / 100 for obj in save_objs]
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=0.02, bases=bases, include_base=True
    )
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=-0.02, bases=bases, include_base=True
    )


def fill_budget_gap(start, stop, step=0.02):
    result = []
    d = start + step
    while d < stop:
        result.append(round(d, 2))
        d += step
    return result


def fill_gaps():
    gaps = []
    # gaps.extend(fill_budget_gap(3.57, 4.03))
    # gaps.extend(fill_budget_gap(3.29, 3.45))
    # gaps.extend(fill_budget_gap(6.92, 7.56))
    # gaps.extend(fill_budget_gap(4.61, 5.00))
    # gaps.extend(fill_budget_gap(5.20, 5.86))
    gaps.extend(fill_budget_gap(6.01, 6.70))
    # gaps.extend([4.39, 4.37])
    print(gaps)
    plan_eval_bit_budgets(
        model_arch="llm", points=0, step=0.02, bases=gaps, include_base=True
    )


if __name__ == "__main__":
    # fill_gaps()
    plan_432_bits()
    plan_567_bits()
