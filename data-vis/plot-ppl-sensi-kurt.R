#!/usr/bin/env Rscript

library(tidyverse)
library(stringr)
library(plyr)
library(dplyr)
library(optparse)
library(openxlsx)

calc_bpp <- function(config) {
  if (config == "base") {
    return(16.0)
  } else if (startsWith(config, "b")) {
    b1 <- strtoi(substr(config, 2, 2))
    g1 <- strtoi(substr(config, 4, nchar(config)))
    b2 <- 8
    g2 <- 128
    return(round(b1 + 2 * b2 / g1 + 32 / g1 / g2, digits = 2))
  } else {
    return(round(as.numeric(sub("_", ".", config)), digits = 2))
  }
}

load_data <- function(baseline_data_dir, mxq_data_dir, type) {
  if (is.null(baseline_data_dir)) {
    dat_dir <- path.expand(paste0("data/", type))
    dat_fps <- dir(
      path = dat_dir,
      pattern = ".*\\.csv$",
      recursive = TRUE,
      full.names = TRUE
    )
  } else {
    # Read csv files under baseline_data_dir, do not recursive into sub-folders
    dat_dir <- path.expand(paste0(baseline_data_dir, "/", type))
    dat_fps <- dir(
      path = dat_dir,
      pattern = ".*\\.csv$",
      recursive = FALSE,
      full.names = TRUE
    )
    mxq_dat_dir <- path.expand(paste0(mxq_data_dir, "/", type))
    mxq_dat_fps <- dir(
      path = mxq_dat_dir,
      pattern = ".*\\.csv$",
      recursive = TRUE,
      full.names = TRUE
    )
    dat_fps <- c(dat_fps, mxq_dat_fps)
  }
  names(dat_fps) <- str_match(dat_fps, "mxq/(.*)/")[, 2]
  df_combined <- ldply(
    dat_fps,
    read.csv,
    stringsAsFactors = FALSE,
    .id = "attempt"
  )
  return(df_combined)
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-m", "--mxq_data_dir"),
  type = "character",
  help = "Data directory of MXQ results organized by attempt",
  metavar = "character"
)
args <- parse_args(parser)

if (is.null(args$baseline_data_dir)) {
  df_combined_stor <- load_data(NULL, NULL, "stor")
  df_combined_ppl <- load_data(NULL, NULL, "ppl")
} else {
  df_combined_stor <- load_data(
    args$baseline_data_dir, args$mxq_data_dir, "stor"
  )
  df_combined_ppl <- load_data(
    args$baseline_data_dir, args$mxq_data_dir, "ppl"
  )
}

df_combined_stor <- df_combined_stor |>
  select(
    model, algo, config, attempt, load_mem_allot, model_storage_size
  ) |>
  mutate(
    load_mem_allot = round(load_mem_allot / 1024^3, digits = 2),
    model_storage_size = round(model_storage_size / 1024^3, digits = 2),
  )

df_combined_ppl <- df_combined_ppl |>
  select(
    model, algo, config, attempt,
    ppl_wikitext, ppl_c4, quant_duration
  )

df_combined <- df_combined_ppl |>
  left_join(df_combined_stor, join_by(model, algo, config, attempt)) |>
  mutate(
    bpp = sapply(config, calc_bpp)
  ) |>
  distinct(model, algo, config, attempt, .keep_all = TRUE) |>
  relocate(bpp, .after = config)
