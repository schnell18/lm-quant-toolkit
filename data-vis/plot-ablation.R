#!/usr/bin/env Rscript

library(tidyverse)
library(dplyr)
library(readr)
library(openxlsx)
library(ggmagnify)
library(optparse)
library(this.path)

# make reference to library function portable
source(file.path(here("functions"), "utils.R"))
source(file.path(here("functions"), "allocation.R"))

# plot_sensi_vs_ablation <- function()

plot_meta <- function(
    model_id,
    the_dataset,
    df_disp,
    df_hqq) {
  min_ppl <- min(df_disp$ppl)
  max_ppl <- max(df_disp$ppl)
  min_ppl <- floor(min_ppl * 10) / 10
  max_ppl <- ceiling(max_ppl * 10) / 10
  step <- round((max_ppl - min_ppl) / 15, digits = 2)

  plt <- ggplot(df_disp, aes(bpp, ppl, color = ablation)) +
    # scale_x_continuous(
    #   limits = c(0, 9.0),
    #   breaks = seq(0, 9.0, 0.50)
    # ) +
    # scale_y_continuous(
    #   limits = c(min_ppl, max_ppl),
    #   breaks = seq(min_ppl, max_ppl, step)
    # ) +
    geom_point(size = 2) +
    labs(x = "% Memory Increment", y = "Perplexity") +
    theme(
      strip.background = element_rect(
        color = "darkgray", fill = "white", linewidth = 1.0, linetype = "solid"
      ),
      strip.text.x = element_text(face = "bold", size = 12),
      strip.text.y = element_text(face = "bold", size = 12),
      legend.position = "bottom"
    ) +
    # guides(
    #   shape = guide_legend(title = legend_title),
    #   color = guide_legend(title = legend_title)
    # ) +
    facet_grid(method ~ model, scales = "free")

  ggsave(
    # paste0("pdfs/ppl-", type, "-", model_id, "-", the_dataset, ".pdf"),
    "pdfs/abc.pdf",
    plot = plt,
    width = 8,
    height = 6,
    dpi = 600
  )
  return(plt)
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--combined_csv_file"),
  type = "character",
  help = "The combined csv file",
  metavar = "character"
)
parser <- add_option(
  parser, c("-a", "--allot_csv_file"),
  type = "character",
  help = "Allocation CSV file",
  metavar = "character"
)
parser <- add_option(
  parser, c("-t", "--type"),
  type = "character",
  help = "Type diagram",
  metavar = "character"
)

args <- parse_args(parser)
if (is.null(args$csv_file)) {
  combined_csv_fp <- "data/combined.csv"
} else {
  combined_csv_fp <- args$combined_csv_file
}
if (is.null(args$csv_file)) {
  allot_cfg_csv_fp <- "data/quant-cfg-allocation.csv"
} else {
  allot_cfg_csv_fp <- args$allot_csv_file
}
if (is.null(args$type)) {
  type <- "sensi-vs-kurt"
} else {
  type <- args$type
}

# TODO: remove debug lines
allot_cfg_csv_fp <- "endeavors/boost/data/quant-cfg-allocation.csv"
combined_csv_fp <- "endeavors/boost/data/combined.csv"

tup <- load_ppl_mem_inc(allot_cfg_csv_fp, combined_csv_fp)
df_ppl_mem_inc <- tup[[1]]
df_hqq <- tup[[2]]

# models <- unique(df_ppl_mem_inc$model)
model_id <- "Llama-2-7B"
plt <- plot_meta(
  model_id,
  "C4",
  df_ppl_mem_inc |> filter(model == model_id & dataset == "C4"),
  df_hqq
)

