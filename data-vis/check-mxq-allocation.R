#!/usr/bin/env Rscript

library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(optparse)
library(openxlsx)

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

parser <- OptionParser()
parser <- add_option(
  parser, c("-f", "--factor"),
  type = "double",
  help = "Factor to apply",
  metavar = "double"
)
parser <- add_option(
  parser, c("-d", "--data_dir"),
  type = "character",
  help = "Data directory of fnorm csv files",
  metavar = "character"
)
parser <- add_option(
  parser, c("-a", "--allot_csv_file"),
  type = "character",
  help = "Allocation CSV file",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt"),
  type = "character",
  help = "attempt",
  metavar = "character"
)

args <- parse_args(parser)

if (is.null(args$data_dir)) {
  fnorm_dir <- "../src/data"
} else {
  fnorm_dir <- args$data_dir
}
if (is.null(args$factor)) {
  factor <- 2.0
} else {
  factor <- args$factor
}
if (is.null(args$allot_csv_file)) {
  allot_csv_file <- "data/allot/mxq/mxq1/quant-cfg-allot-mxq1.csv"
} else {
  allot_csv_file <- args$allot_csv_file
}
if (is.null(args$attempt)) {
  the_attempt <- "mxq1"
} else {
  the_attempt <- args$attempt
}

fnorm_dir <- path.expand(fnorm_dir)
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = "fnorm-.*\\.csv$",
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- ldply(fnorm_fps, read.csv, stringsAsFactors = FALSE, .id = "model")

k_cols <- c(
  "model",
  "module",
  "layer",
  "cfg",
  "nbit1",
  "gsize1",
  "nbit2",
  "gsize2",
  "fnorm",
  "memmb",
  "params",
  "sensitivity",
  "kurtosis"
)
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

df_sd_mu <- df_fnorm |>
  group_by(model) |>
  dplyr::summarise(
    sigma = sd(sensitivity),
    mu = mean(sensitivity),
    tot_params = sum(params),
  )

df_kurt_scaled <- df_fnorm |>
  group_by(model, module) |>
  dplyr::summarise(
    min_kurt = min(kurtosis),
    max_kurt = max(kurtosis)
  )

df_fnorm <- df_fnorm |>
  left_join(df_sd_mu, by = c("model")) |>
  mutate(
    bpp = nbit1 + 2 * nbit2 / gsize1 + 32 / gsize1 / gsize2
  ) |>
  mutate(
    factor_sensi = ifelse((sensitivity - mu) / sigma > 3, factor, 1)
  ) |>
  mutate(
    cost_sensi = factor_sensi * 100 * 12 * (params / tot_params) / bpp
  ) |>
  left_join(df_kurt_scaled, by = c("model", "module")) |>
  mutate(
    kurt_scaled = (kurtosis - min_kurt) / (max_kurt - min_kurt),
    cost_kurt = kurt_scaled * 100 * 12 * (params / tot_params) / bpp
  )

by <- join_by(model == model, module == module, layer == layer, cfg == cfg)
df_cfgs <- read_csv(allot_csv_file)

if ("attempt" %in% names(df_cfgs)) {
  df_cfgs <- df_cfgs |>
    filter(attempt == the_attempt)
}

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
    memmb = sum(memmb),
    memmb_base = sum(memmb_base),
    fnorm = sum(fnorm),
    fnorm_base = sum(fnorm_base),
    cost_sensi = sum(cost_sensi),
    cost_kurt = sum(cost_kurt),
    cost_sensi_base = sum(cost_sensi_base),
    cost_kurt_base = sum(cost_kurt_base),
    params_tot = sum(params)
  ) |>
  mutate(
    memmb = round(memmb, digits = 4),
    memmb_base = round(memmb_base, digits = 4),
    fnorm = round(fnorm, digits = 4),
    fnorm_base = round(fnorm_base, digits = 4),
    cost_sensi = round(cost_sensi, digits = 4),
    cost_kurt = round(cost_kurt, digits = 4),
    cost_sensi_base = round(cost_sensi_base, digits = 4),
    cost_kurt_base = round(cost_kurt_base, digits = 4),
    fnorm_imporved = fnorm < fnorm_base,
    theory_memmb = params_tot * bit_budget / 8 / 1024^2,
    mem_pct_of_base = round(100 * memmb / memmb_base, digits = 4),
    mem_pct_of_theory = round(100 * memmb / theory_memmb, digits = 4)
  ) |>
  select(
    c(
      "model",
      "cfg_base",
      "bit_budget",
      "cost_sensi_base",
      "cost_sensi",
      "cost_kurt_base",
      "cost_kurt",
      "fnorm_base",
      "fnorm",
      "fnorm_imporved",
      "mem_pct_of_base",
      "mem_pct_of_theory"
    )
  )

write.xlsx(
  df_check_sum,
  "allot-check.xlsx",
  overwrite = TRUE,
  asTable = TRUE
)
