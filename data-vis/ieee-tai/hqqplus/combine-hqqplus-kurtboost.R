#!/usr/bin/env Rscript

library(tidyverse)
library(stringr)
library(optparse)

# Bits-per-param for an HQQ(+) config. FP16 -> 16. For bXgY backbones the
# scalar/zero overhead assumes 8-bit, group-128 meta (matches combine-qwen.R).
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

# Columns produced by the HQQ+ SFT bench (src/lm_quant_toolkit/eval/bench_sft.py).
llm_cols <- c(
  "model", "algorithm", "nbits", "group_size",
  "boost_stop", "top_m", "n_boosted_pairs",
  "ppl_wikitext", "ppl_mem_allot"
)

read_one <- function(path) {
  read_csv(path, col_select = all_of(llm_cols), show_col_types = FALSE) |>
    mutate(across(c(nbits, group_size, boost_stop, top_m), as.integer))
}

# Method label: FP16 / HQQ+ baselines, or KB<stop><top_m> for KurtBoost.
derive_method <- function(algorithm, boost_stop, top_m) {
  dplyr::case_when(
    algorithm == "fp16" ~ "FP16",
    algorithm == "HQQ+" ~ "HQQ+",
    algorithm == "kurtboost" ~ paste0("KB", boost_stop, top_m),
    TRUE ~ algorithm
  )
}

# Config label: base for FP16, else bXgY backbone (e.g. b1g8, b2g8).
derive_config <- function(algorithm, nbits, group_size) {
  ifelse(
    algorithm == "fp16",
    "base",
    paste0("b", nbits, "g", group_size)
  )
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-r", "--round_dir"),
  type = "character",
  help = "Round directory holding baselines/ and kurtboost/ result CSVs",
  metavar = "character"
)
parser <- add_option(
  parser, c("-o", "--output"),
  type = "character",
  help = "Output combined CSV path",
  metavar = "character"
)
args <- parse_args(parser)

round_dir <- if (is.null(args$round_dir)) "rounds/round1" else args$round_dir
out_fp <- if (is.null(args$output)) "combined-kurtboost.csv" else args$output

csvs <- list.files(
  c(file.path(round_dir, "baselines"), file.path(round_dir, "kurtboost")),
  pattern = "\\.csv$",
  full.names = TRUE
)
if (length(csvs) == 0) {
  stop(sprintf("No result CSVs found under %s/{baselines,kurtboost}", round_dir))
}

combined <- bind_rows(lapply(csvs, read_one)) |>
  mutate(
    method = derive_method(algorithm, boost_stop, top_m),
    config = derive_config(algorithm, nbits, group_size),
    bpp = sapply(config, calc_bpp),
    memory = round(ppl_mem_allot / 1024^3, digits = 2)
  ) |>
  select(model, method, config, bpp, ppl_wikitext, memory) |>
  distinct(model, method, config, .keep_all = TRUE) |>
  arrange(model, config, method)

write_csv(combined, out_fp, na = "")
cat(sprintf("Wrote %d rows to %s\n", nrow(combined), out_fp))
