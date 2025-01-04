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
  parser, c("-c", "--milp_cost_csv"),
  type = "character",
  help = "Dump of MiLP cost csv from the HQQ",
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

if (is.null(args$milp_cost_csv)) {
  milp_cost_csv <- "debug.csv"
} else {
  milp_cost_csv <- args$milp_cost_csv
}
if (is.null(args$attempt)) {
  the_attempt <- "mxq1"
} else {
  the_attempt <- args$attempt
}
if (is.null(args$allot_csv_file)) {
  allot_csv_file <- paste0(
    "data/allot/mxq/", the_attempt, "/quant-allot-", the_attempt, ".csv"
  )
} else {
  allot_csv_file <- args$allot_csv_file
}

df_fnorm <- read_csv(milp_cost_csv)

k_cols <- c(
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
  "sensitivity"
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

by <- join_by(module == module, layer == layer, cfg == cfg)
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
    suffix = c("", "_hqq"),
    join_by(
      module == module,
      layer == layer,
      cfg_base == cfg
    )
  )
df_check_det <- df_check_sum |>
  select(
    !c(
      "param_cnt",
      "params_quant_tot",
      "params_hqq",
      "sensitivity_hqq",
      "nbit1",
      "nbit2",
      "gsize1",
      "gsize2",
      "nbit1_hqq",
      "nbit2_hqq",
      "gsize1_hqq",
      "gsize2_hqq",
    )
  )
write.xlsx(
  df_check_det,
  "allot-check-det.xlsx",
  overwrite = TRUE,
  asTable = TRUE
)

df_final <- df_check_sum |>
  group_by(cfg_base, bit_budget) |>
  dplyr::summarise(
    memmb = sum(memmb),
    memmb_hqq = sum(memmb_hqq),
    fnorm = sum(fnorm),
    fnorm_hqq = sum(fnorm_hqq),
    params_tot = sum(params)
  ) |>
  mutate(
    memmb = round(memmb, digits = 4),
    memmb_hqq = round(memmb_hqq, digits = 4),
    fnorm = round(fnorm, digits = 4),
    fnorm_hqq = round(fnorm_hqq, digits = 4),
    fnorm_imporved = fnorm < fnorm_hqq,
    theory_memmb = params_tot * bit_budget / 8 / 1024^2,
    mem_pct_of_hqq = round(100 * memmb / memmb_hqq, digits = 4),
    mem_pct_of_theory = round(100 * memmb / theory_memmb, digits = 4)
  ) |>
  select(
    c(
      "cfg_base",
      "bit_budget",
      "fnorm_hqq",
      "fnorm",
      "fnorm_imporved",
      "memmb_hqq",
      "memmb",
      "mem_pct_of_hqq",
      "mem_pct_of_theory"
    )
  )

write.xlsx(
  df_final,
  "df_final.xlsx",
  overwrite = TRUE,
  asTable = TRUE
)
