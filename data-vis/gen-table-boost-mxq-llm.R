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

sb_levels <- c(
  "mxq-SB20", "mxq-SB21", "mxq-SB22", "mxq-SB23",
  "mxq-SB30", "mxq-SB31", "mxq-SB32", "mxq-SB33"
)

sb_labels <- c(
  "MXQ-SB20", "MXQ-SB21", "MXQ-SB22", "MXQ-SB23",
  "MXQ-SB30", "MXQ-SB31", "MXQ-SB32", "MXQ-SB33"
)

sb_ab_levels <- c(
  "mxq-SBAB20", "mxq-SBAB21", "mxq-SBAB22", "mxq-SBAB23",
  "mxq-SBAB30", "mxq-SBAB31", "mxq-SBAB32", "mxq-SBAB33"
)

sb_ab_labels <- c(
  "MXQ-SBAB20", "MXQ-SBAB21", "MXQ-SBAB22", "MXQ-SBAB23",
  "MXQ-SBAB30", "MXQ-SBAB31", "MXQ-SBAB32", "MXQ-SBAB33"
)

kb_levels <- c(
  "mxq-KB20", "mxq-KB21", "mxq-KB22", "mxq-KB23",
  "mxq-KB30", "mxq-KB31", "mxq-KB32", "mxq-KB33"
)

kb_labels <- c(
  "MXQ-KB20", "MXQ-KB21", "MXQ-KB22", "MXQ-KB23",
  "MXQ-KB30", "MXQ-KB31", "MXQ-KB32", "MXQ-KB33"
)

kb_ab_levels <- c(
  "mxq-KBAB20", "mxq-KBAB21", "mxq-KBAB22", "mxq-KBAB23",
  "mxq-KBAB30", "mxq-KBAB31", "mxq-KBAB32", "mxq-KBAB33"
)

kb_ab_labels <- c(
  "MXQ-KBAB20", "MXQ-KBAB21", "MXQ-KBAB22", "MXQ-KBAB23",
  "MXQ-KBAB30", "MXQ-KBAB31", "MXQ-KBAB32", "MXQ-KBAB33"
)

if (comparison == "sensi-vs-kurt") {
  df_no_abl <- read_csv(csv_fp) |>
    filter(
      !grepl("-abl", attempt)
    )
  level_pairs <- zipcat(sb_levels, kb_levels)
  label_pairs <- zipcat(sb_labels, kb_labels)
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
} else if (comparison == "sensi-vs-ablation") {
  df_no_kurt <- read_csv(csv_fp) |>
    filter(
      !grepl("kurt", attempt) & algo == "mxq" & attempt != "mxq1"
    )
  df_latex <- process_dataframe(
    df_no_kurt,
    zipcat(sb_levels, sb_ab_levels),
    zipcat(sb_labels, sb_ab_labels)
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
} else if (comparison == "kurt-vs-ablation") {
  df_no_sensi <- read_csv(csv_fp) |>
    filter(
      !grepl("sensi", attempt) & algo == "mxq" & attempt != "mxq1"
    )
  df_latex <- process_dataframe(
    df_no_sensi,
    zipcat(kb_levels, kb_ab_levels),
    zipcat(kb_labels, kb_ab_labels)
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
}
