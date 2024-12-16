#!/usr/bin/env Rscript

library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(optparse)

save_fnorm_by_model <- function(df, factor) {
  models <- unique(df$model)
  for (mdl in models) {
    df_by_model <- df |>
      filter(model == mdl) |>
      select(-c("model"))
    sd <- sd(df_by_model$kurtosis_scaled)
    mu <- mean(df_by_model$kurtosis_scaled)
    df_by_model <- df_by_model |>
      mutate(
        kurtosis_scaled = ifelse(abs(kurtosis_scaled - mu) / sd > 3, factor, 1)
      ) |>
      write_csv(paste0("fnorm-", mdl, ".csv"))
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
fnorm_dir <- "/tmp/apportion-sensi"
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

# df_fnorm <- df_fnorm |>
#   select(!c("kurtosis_scaled"))

save_fnorm_by_model(df_fnorm, factor)
