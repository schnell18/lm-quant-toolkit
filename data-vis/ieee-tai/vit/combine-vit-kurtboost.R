#!/usr/bin/env Rscript

library(tidyverse)
library(stringr)

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

extract_attempt <- function(path) {
  base <- basename(path)
  m <- regmatches(
    base,
    regexec(
      "result-(kurt-boost(?:-ablation)?-[0-9]+-[0-9]+)-[0-9]+\\.csv$",
      base,
      perl = TRUE
    )
  )[[1]]
  if (length(m) == 2) {
    return(m[2])
  }
  return(NA_character_)
}

zs_cols <- c(
  "model", "algo", "config",
  "zeroshot_mem_allot", "zeroshot_mem_reserved",
  "acc1_zeroshot_cls", "acc5_zeroshot_cls",
  "recall_zeroshot_cls", "duration_zeroshot_cls"
)

read_one <- function(path, attempt) {
  read_csv(path, col_select = all_of(zs_cols), show_col_types = FALSE) |>
    mutate(
      attempt = attempt,
      zeroshot_mem_allot = zeroshot_mem_allot / 1024 / 1024,
      zeroshot_mem_reserved = zeroshot_mem_reserved / 1024 / 1024
    )
}

baseline_dir <- "baselines"
baseline_csvs <- list.files(
  baseline_dir,
  pattern = "^result-baseline-.*\\.csv$",
  full.names = TRUE
)

kb_dir <- "kurtboost"
kb_csvs <- list.files(
  kb_dir,
  pattern = "^result-kurt-boost-[0-9]+-[0-9]+-.*\\.csv$",
  full.names = TRUE
)

abl_dir <- "ablation"
abl_csvs <- list.files(
  abl_dir,
  pattern = "^result-kurt-boost-ablation-[0-9]+-[0-9]+-.*\\.csv$",
  full.names = TRUE
)

baseline_list <- lapply(baseline_csvs, read_one, attempt = "baseline")
kb_list <- lapply(kb_csvs, function(p) read_one(p, attempt = extract_attempt(p)))
abl_list <- lapply(abl_csvs, function(p) read_one(p, attempt = extract_attempt(p)))

combined <- bind_rows(c(baseline_list, kb_list, abl_list)) |>
  mutate(bpp = sapply(config, calc_bpp)) |>
  relocate(attempt, .after = algo) |>
  relocate(bpp, .after = config) |>
  arrange(model, algo, attempt, config)

write_csv(combined, "combined-kurtboost.csv", na = "")
