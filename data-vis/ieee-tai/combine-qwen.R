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

load_data <- function(baseline_data_dir, type) {
  # Read csv files under baseline_data_dir, do not recursive into sub-folders
  dat_dir <- path.expand(paste0(baseline_data_dir, "/", type))
  dat_fps <- dir(
    path = dat_dir,
    pattern = ".*\\.csv$",
    recursive = FALSE,
    full.names = TRUE
  )

  names(dat_fps) <- str_match(dat_fps, "ours/(.*)/")[, 2]
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
  parser, c("-b", "--data_dir"),
  type = "character",
  help = "Data directory of baseline results",
  metavar = "character"
)

args <- parse_args(parser)

# if (is.null(args$data_dir)) {
#   $data_dir = "data"
# }

df_combined_stor <- load_data("data", "stor")
df_combined_ppl <- load_data("data", "ppl")
df_combined_qnt <- load_data("data", "qnt")
# df_combined_allot <- load_data(
#   args$data_dir, args$mxq_data_dir, "allot"
# )
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

df_combined_qnt <- df_combined_qnt |>
  select(
    model, algo, config, attempt, quant_duration
  )

df_model_params <- tribble(
  ~model, ~param_count,
  "Qwen3.5-2B", 2000000000,
  "Qwen3.5-4B", 4000000000,
  "Qwen3.5-9B", 9000000000
)

df_combined <- df_combined_ppl |>
  left_join(df_combined_qnt, join_by(model, algo, config, attempt)) |>
  left_join(df_combined_stor, join_by(model, algo, config, attempt)) |>
  mutate(
    quant_duration =
      ifelse(quant_duration.x > 0, quant_duration.x, quant_duration.y)
  ) |>
  left_join(df_model_params, join_by(model)) |>
  mutate(
    bpp = sapply(config, calc_bpp)
  ) |>
  select(!c("quant_duration.x", "quant_duration.y")) |>
  distinct(model, algo, config, attempt, .keep_all = TRUE) |>
  relocate(bpp, .after = config)

write_csv(df_combined, "data/combined.csv", na = "")
write.xlsx(
  df_combined, "data/combined.xlsx",
  overwrite = TRUE, asTable = TRUE
)
