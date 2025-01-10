#!/usr/bin/env Rscript

library(tidyverse)
library(openxlsx)
library(knitr)
library(kableExtra)
library(optparse)
library(this.path)

# make reference to library function portable
source(file.path(here("functions"), "utils.R"))
source(file.path(here("functions"), "tabular.R"))

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--csv_file"),
  type = "character",
  help = "The combined csv file",
  metavar = "character"
)
parser <- add_option(
  parser, c("--comparison"),
  type = "character",
  help = "The comparison of metrics to present",
  metavar = "character"
)
parser <- add_option(
  parser, c("--experiment"),
  type = "character",
  help = "The experiment to include in the table caption",
  metavar = "character"
)

args <- parse_args(parser)
if (is.null(args$csv_file)) {
  csv_fp <- "data/combined.csv"
} else {
  csv_fp <- args$csv_file
}
if (is.null(args$experiment)) {
  experiment <- "experiment x"
} else {
  experiment <- args$experiment
}
if (is.null(args$comparison)) {
  comparison <- "sensi-vs-kurt"
} else {
  comparison <- args$comparison
}

baseline_levels <- c("hqq", "fp16", "awq", "gptq", "bnb")
baseline_labels <- c("HQQ", "FP16", "AWQ", "GPTQ", "BnB")

sm_levels <- c("mxq-SM1", "mxq-SM2", "mxq-SM3")
sm_labels <- c("MXQ-SM1", "MXQ-SM2", "MXQ-SM3")
km_levels <- c("mxq-KM1", "mxq-KM2", "mxq-KM3")
km_labels <- c("MXQ-KM1", "MXQ-KM2", "MXQ-KM3")

sm_ab_levels <- c("mxq-SMAB")
sm_ab_labels <- c("MXQ-SMAB")
km_ab_levels <- c("mxq-KMAB")
km_ab_labels <- c("MXQ-KMAB")

if (comparison == "sensi-vs-kurt") {
  df_no_abl <- read_csv(csv_fp) |>
    filter(
      !grepl("-abl", attempt)
    )
  level_pairs <- zipcat(sm_levels, km_levels)
  label_pairs <- zipcat(sm_labels, km_labels)

  level_pairs <- zipcat(sm_levels, km_levels)
  label_pairs <- zipcat(sm_labels, km_labels)
  df_latex <- process_dataframe(
    df_no_abl,
    c(level_pairs, baseline_levels, "mxq"),
    c(label_pairs, baseline_labels, "MXQ")
  )
  df_latex_4bit <- df_latex |> filter(bpp == 4.13 | bpp == 4.25 | bpp == 4.51)
  dump_latex_table(
    df_latex_4bit,
    paste0(experiment, " (4-bit)"),
    paste0(comparison, "-4bit.tex")
  )
  df_latex_3bit <- df_latex |> filter(bpp == 3.13 | bpp == 3.25 | bpp == 3.51)
  dump_latex_table(
    df_latex_3bit,
    paste0(experiment, " (3-bit)"),
    paste0(comparison, "-3bit.tex")
  )
  df_latex_others <- df_latex |>
    filter(
      bpp != 3.13 & bpp != 3.25 & bpp != 3.51 &
        bpp != 4.13 & bpp != 4.25 & bpp != 4.51
    )
  dump_latex_table(
    df_latex_others,
    paste0(experiment, " (other-bit)"),
    paste0(comparison, "-others.tex")
  )
} else if (comparison == "sensi-vs-ablation") {
  df_no_kurt <- read_csv(csv_fp) |>
    filter(
      !grepl("kurt", attempt) & attempt != "mxq1"
    ) |>
    filter(algo == "mxq" | algo == "hqq")
  df_latex <- process_dataframe(
    df_no_kurt,
    c(sm_levels, baseline_levels, sm_ab_levels, "mxq"),
    c(sm_labels, baseline_labels, sm_ab_labels, "MXQ")
  )
  dump_latex_table(
    df_latex,
    experiment,
    paste0(comparison, ".tex")
  )
} else if (comparison == "kurt-vs-ablation") {
  df_no_sensi <- read_csv(csv_fp) |>
    filter(
      !grepl("sensi", attempt) & attempt != "mxq1"
    ) |>
    filter(algo == "mxq" | algo == "hqq")
  df_latex <- process_dataframe(
    df_no_sensi,
    c(km_levels, baseline_levels, km_ab_levels, "mxq"),
    c(km_labels, baseline_labels, km_ab_labels, "MXQ")
  )
  dump_latex_table(
    df_latex,
    experiment,
    paste0(comparison, ".tex")
  )
}
