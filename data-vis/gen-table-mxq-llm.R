#!/usr/bin/env Rscript

library(tidyverse)
library(openxlsx)
library(knitr)
library(kableExtra)
library(optparse)
library(this.path)

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
  parser, c("--attempt"),
  type = "character",
  help = "The attempt to plot",
  metavar = "character"
)

args <- parse_args(parser)
if (is.null(args$csv_file)) {
  csv_fp <- "data/combined.csv"
} else {
  csv_fp <- args$csv_file
}
if (is.null(args$attempt)) {
  the_attempt <- "mxq1"
} else {
  the_attempt <- args$attempt
}

df_all <- read_csv(csv_fp)
levels <- c("mxq", "hqq", "fp16", "awq", "gptq", "bnb")
labels <- c("MXQ", "HQQ", "FP16", "AWQ", "GPTQ", "BnB")
df_latex <- process_dataframe(df_all, levels, labels)
dump_latex_table(df_latex, the_attempt)
