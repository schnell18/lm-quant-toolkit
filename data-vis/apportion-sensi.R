#!/usr/bin/env Rscript

library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(openxlsx)
library(optparse)

save_fnorm_by_model <- function(df, factor, combine_only) {
  models <- unique(df$model)
  for (mdl in models) {
    df_by_model <- df |>
      filter(model == mdl) |>
      select(-c("model"))
    if (combine_only) {
      df_by_model <- df_by_model |>
        dplyr::rename(
          sensitivity = sensi
        ) |>
        write_csv(paste0("fnorm-", mdl, ".csv"))
    } else {
      sd <- sd(df_by_model$sensi)
      mu <- mean(df_by_model$sensi)
      df_by_model <- df_by_model |>
        mutate(
          sensitivity = ifelse(abs(sensi - mu) / sd > 3, factor, 1)
        ) |>
        select(!c("sensi")) |>
        write_csv(paste0("fnorm-", mdl, ".csv"))
    }
  }
}

map_to_part <- function(module) {
  if (module == "self_attn.q_proj" || module == "self_attn.k_proj" ||
    module == "self_attn.v_proj") {
    return("attn_in")
  } else if (module == "self_attn.o_proj") {
    return("attn_out")
  } else if (module == "mlp.gate_proj" || module == "mlp.up_proj") {
    return("mlp_gate")
  } else if (module == "mlp.down_proj") {
    return("mlp_down")
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
  parser, c("-c", "--combine_only"),
  type = "logical",
  action = "store_true",
  help = "Merge sensitivity data only",
  metavar = "logical"
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
if (is.null(args$combine_only)) {
  combine_only <- FALSE
} else {
  combine_only <- args$combine_only
}

fnorm_dir <- path.expand(fnorm_dir)
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = "fnorm-.*\\.csv$",
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- plyr::ldply(
  fnorm_fps,
  read.csv,
  stringsAsFactors = FALSE,
  .id = "model"
)
df_fnorm <- df_fnorm |>
  mutate(
    part = sapply(module, map_to_part)
  ) |>
  select(!c("sensitivity"))

df_sensi <- read_csv("data/llama-sensitivity.csv") |>
  filter(
    dataset == "pileval" & nbits == 4 & group_size == 64
  ) |>
  dplyr::rename(
    sensi = sensitivity
  )

df_com1 <- df_fnorm |>
  filter(
    model != "Meta-Llama-3-8B"
  ) |>
  left_join(
    df_sensi,
    by = join_by(
      model == model,
      layer == layer,
      part == part
    )
  ) |>
  select(-c("dataset", "part", "nbits", "group_size"))

df_com2_1 <- df_fnorm |>
  filter(
    model == "Meta-Llama-3-8B" & part != "attn_out"
  ) |>
  left_join(
    df_sensi,
    by = join_by(
      model == model,
      part == part,
      layer == layer
    )
  ) |>
  select(-c("dataset", "part", "nbits", "group_size"))

# The Llama3-8B has no attn_out metric, use attn_in instead
df_sensi_attn_in <- df_sensi |>
  filter(part == "attn_in")

df_com2_2 <- df_fnorm |>
  filter(
    model == "Meta-Llama-3-8B" & part == "attn_out"
  ) |>
  left_join(
    df_sensi_attn_in,
    by = join_by(
      model == model,
      layer == layer
    )
  ) |>
  select(-c("dataset", "part.x", "part.y", "nbits", "group_size"))


df_com <- bind_rows(df_com1, df_com2_1, df_com2_2)
save_fnorm_by_model(df_com, factor, combine_only)
