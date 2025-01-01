calc_mem_inc <- function(df_cfgs) {
  df_cfg_mem <- df_cfgs |>
    dplyr::mutate(
      base_cfg = sapply(bit_budget, budget_to_cfg),
      cfg = paste0("b", b1, "g", g1),
      bpp = sapply(cfg, calc_bpp),
      base_bpp = sapply(base_cfg, calc_bpp),
      mem_orig = param_cnt * base_bpp,
      mem_new = param_cnt * bpp
    ) |>
    select(-c("b1", "g1", "b2", "g2", "base_bpp", "base_cfg", "memmb")) |>
    dplyr::mutate(
      cfg = factor(
        cfg,
        levels = c(
          "b2g128", "b2g64", "b2g32",
          "b3g128", "b3g64", "b3g32",
          "b4g128", "b4g64", "b4g32",
          "b8g128", "b8g64", "b8g32"
        )
      )
    ) |>
    group_by(attempt, model, bit_budget) |>
    summarise(
      mem_orig = sum(mem_orig),
      mem_new = sum(mem_new)
    ) |>
    mutate(
      mem_orig = mem_orig / 8 / 1024^2,
      mem_new = mem_new / 8 / 1024^2,
      increment = round(100 * (mem_new - mem_orig) / mem_orig, digits = 2)
    )
  return(df_cfg_mem)
}

load_ppl_mem_inc <- function(allot_cfg_csv_fp, combined_csv_fp) {
  df_cfgs <- read_csv(allot_cfg_csv_fp) |>
    mutate(
      model = factor(
        model,
        levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
        labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
      )
    )

  df_w_base <- read_csv(combined_csv_fp) |>
    mutate(
      model = factor(
        model,
        levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
        labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
      )
    )

  df_hqq <- df_w_base |>
    filter(
      algo == "hqq"
    )

  df_cfg_mem <- calc_mem_inc(df_cfgs)

  df_ppl_mem_inc <- df_cfg_mem |>
    left_join(
      df_hqq,
      suffix = c("", "_h"),
      by = join_by(model, bit_budget == bpp)
    ) |>
    left_join(
      df_w_base,
      suffix = c("_hqq", ""),
      by = join_by(model, attempt, bit_budget == bpp)
    ) |>
    mutate(
      ppl_wikitext_decr = round(
        100 * (ppl_wikitext_hqq - ppl_wikitext) / ppl_wikitext_hqq,
        digits = 2
      ),
      ppl_c4_decr = round(
        100 * (ppl_c4_hqq - ppl_c4) / ppl_c4_hqq,
        digits = 2
      ),
      mem_incr = round(
        100 * (load_mem_allot - load_mem_allot_hqq) / load_mem_allot_hqq,
        digits = 2
      )
    ) |>
    rename(bpp = bit_budget) |>
    select(
      c(
        "model",
        "attempt",
        "bpp",
        "increment",
        "ppl_wikitext_decr",
        "ppl_c4_decr",
        "mem_incr",
        "ppl_wikitext",
        "ppl_c4",
        "ppl_wikitext_hqq",
        "ppl_c4_hqq",
        "mem_orig",
        "mem_new",
        "load_mem_allot",
        "load_mem_allot_hqq"
      )
    )


  df_ppl_mem_inc <- df_ppl_mem_inc |>
    filter(
      !is.na(attempt) & attempt != "mxq1"
    ) |>
    separate_wider_regex(
      attempt,
      c(method = "\\w+-\\w+", "-", stop_topm = "\\d-\\d")
    ) |>
    mutate(
      ablation = ifelse(grepl("-abl", method), TRUE, FALSE),
      method1 = ifelse(grepl("sensi-", method), "SensiBoost", "KurtBoost")
    ) |>
    mutate(method = method1) |>
    select(!c("method1")) |>
    pivot_longer(
      cols = c("ppl_wikitext", "ppl_c4"),
      names_to = c(".value", "dataset"),
      names_sep = "_"
    ) |>
    mutate(
      dataset = factor(
        dataset,
        levels = c("wikitext", "c4"),
        labels = c("WikiText2", "C4")
      )
    )
  list(df_ppl_mem_inc, df_hqq)
}
