#!/usr/bin/env Rscript

library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(optparse)

budget_to_cfg <- function(budget) {
  if (budget == 3.13) {
    return("b3g128")
  } else if (budget == 3.25) {
    return("b3g64")
  } else if (budget == 3.51) {
    return("b3g32")
  } else if (budget == 4.13) {
    return("b4g128")
  } else if (budget == 4.25) {
    return("b4g64")
  } else if (budget == 4.51) {
    return("b4g32")
  } else if (budget == 8.13) {
    return("b8g128")
  } else if (budget == 8.25) {
    return("b8g64")
  } else if (budget == 8.51) {
    return("b8g32")
  } else if (budget == 2.13) {
    return("b2g128")
  } else if (budget == 2.25) {
    return("b2g64")
  } else if (budget == 2.51) {
    return("b2g32")
  } else {
    return("b4g64")
  }
}

strip_name <- function(name) {
  start <- nchar("fnorm-") + 1
  stop <- nchar(name) - 4
  return(substr(name, start, stop))
}

fnorm_dir <- path.expand("data/fnorm")
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = "fnorm-.*\\.csv$",
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- ldply(fnorm_fps, read.csv, stringsAsFactors = FALSE, .id = "model")

k_cols <- c("model", "module", "layer", "cfg", "fnorm", "memmb", "params")
df_fnorm <- df_fnorm |>
  mutate(
    cfg = paste0("b", nbit1, "g", gsize1)
  ) |>
  select(all_of(k_cols)) |>
  mutate(
    cfg = factor(
      cfg,
      levels = c(
        "b2g128", "b2g64", "b2g32",
        "b3g128", "b3g64", "b3g32",
        "b4g128", "b4g64", "b4g32",
        "b8g128", "b8g64", "b8g32"
      )
    )
  )

by <- join_by(model == model, module == module, layer == layer, cfg == cfg)
df_cfgs <- read_csv("data/allot/mxq-quant-cfgs-mxq1.csv")
df_check <- df_cfgs |>
  mutate(
    cfg = paste0("b", b1, "g", g1),
    cfg_base = sapply(bit_budget, budget_to_cfg)
  ) |>
  select(-c("b1", "g1", "b2", "g2", "memmb")) |>
  mutate(
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
  left_join(df_fnorm, by)

df_check_sum <- df_check |>
  left_join(
    df_fnorm,
    suffix = c("", "_base"),
    join_by(
      model == model,
      module == module,
      layer == layer,
      cfg_base == cfg
    )
  ) |>
  group_by(model, cfg_base, bit_budget) |>
  dplyr::summarise(
    fnorm = sum(fnorm),
    fnorm_base = sum(fnorm_base),
    memmb = sum(memmb),
    memmb_base = sum(memmb_base),
    params_tot = sum(params)
  ) |>
  mutate(
    theory_memmb = params_tot * bit_budget / 8 / 1024^2,
    fnorm_imporved = fnorm < fnorm_base,
    mem_improved = memmb < memmb_base
  ) |>
  mutate(
    within_theory_bound = memmb < theory_memmb
  )
