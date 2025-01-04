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

sm_levels <- c("mxq-SM2")
sm_labels <- c("MXQ-SM2")
km_levels <- c("mxq-KM")
km_labels <- c("MXQ-KM")

if (comparison == "sensi-vs-kurt") {
  df_all <- read_csv(csv_fp)
  level_pairs <- zipcat(sm_levels, km_levels)
  label_pairs <- zipcat(sm_labels, km_labels)

  df_latex <- process_dataframe(
    df_all,
    c(level_pairs, baseline_levels, "mxq"),
    c(label_pairs, baseline_labels, "MXQ")
  )

  dump_latex_table(
    df_latex,
    paste0(experiment),
    paste0(comparison, ".tex")
  )
}
