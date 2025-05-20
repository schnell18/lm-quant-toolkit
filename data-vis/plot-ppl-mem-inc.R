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

plot_sensi_vs_ablation <- function(
    model_id,
    the_dataset,
    df_ppl_mem_inc,
    df_hqq,
    mag_from,
    mag_to,
    low_bound_scale = 0.99) {
  df_disp <- df_ppl_mem_inc |>
    filter(model == model_id & dataset == the_dataset) |>
    filter(method == "SensiBoost")
  plt <- plot_meta(
    aes(x = increment, y = ppl, shape = ablation, color = ablation),
    "sbab", model_id, the_dataset, df_disp, df_hqq, mag_from, mag_to,
    low_bound_scale
  )
  return(plt)
}

plot_kurt_vs_ablation <- function(
    model_id,
    the_dataset,
    df_ppl_mem_inc,
    df_hqq,
    mag_from,
    mag_to,
    low_bound_scale = 0.99) {
  df_disp <- df_ppl_mem_inc |>
    filter(model == model_id & dataset == the_dataset) |>
    filter(method == "KurtBoost")
  plt <- plot_meta(
    aes(x = increment, y = ppl, shape = ablation, color = ablation),
    "kbab", model_id, the_dataset, df_disp, df_hqq, mag_from, mag_to,
    low_bound_scale
  )
  return(plt)
}

plot_sensi_vs_kurt <- function(
    model_id,
    the_dataset,
    df_ppl_mem_inc,
    df_hqq,
    mag_from,
    mag_to,
    low_bound_scale = 0.99) {
  df_disp <- df_ppl_mem_inc |>
    filter(model == model_id & dataset == the_dataset) |>
    filter(ablation == FALSE)

  plt <- plot_meta(
    aes(x = increment, y = ppl, shape = method, color = method),
    "sk", model_id, the_dataset, df_disp, df_hqq, mag_from, mag_to,
    low_bound_scale
  )
  return(plt)
}

plot_meta <- function(
    ass,
    type,
    model_id,
    the_dataset,
    df_disp,
    df_hqq,
    mag_from,
    mag_to,
    low_bound_scale = 0.99) {
  df_hqq_mark <- df_hqq |> filter(model == model_id & bpp < 8)
  line_hqq_451 <- df_hqq_mark |> filter(bpp == 4.51)
  line_hqq_425 <- df_hqq_mark |> filter(bpp == 4.25)
  line_hqq_413 <- df_hqq_mark |> filter(bpp == 4.13)
  line_hqq_351 <- df_hqq_mark |> filter(bpp == 3.51)
  line_hqq_325 <- df_hqq_mark |> filter(bpp == 3.25)
  line_hqq_313 <- df_hqq_mark |> filter(bpp == 3.13)
  baseline_451 <- ifelse(
    the_dataset == "WikiText2", line_hqq_451$ppl_wikitext, line_hqq_451$ppl_c4
  )
  baseline_425 <- ifelse(
    the_dataset == "WikiText2", line_hqq_425$ppl_wikitext, line_hqq_425$ppl_c4
  )
  baseline_413 <- ifelse(
    the_dataset == "WikiText2", line_hqq_413$ppl_wikitext, line_hqq_413$ppl_c4
  )
  baseline_351 <- ifelse(
    the_dataset == "WikiText2", line_hqq_351$ppl_wikitext, line_hqq_351$ppl_c4
  )
  baseline_325 <- ifelse(
    the_dataset == "WikiText2", line_hqq_325$ppl_wikitext, line_hqq_325$ppl_c4
  )
  baseline_313 <- ifelse(
    the_dataset == "WikiText2", line_hqq_313$ppl_wikitext, line_hqq_313$ppl_c4
  )

  min_ppl <- min(df_disp$ppl) * low_bound_scale
  max_ppl <- max(df_disp$ppl)
  max_hqq_ppl <- ifelse(
    the_dataset == "WikiText2",
    max(df_hqq_mark$ppl_wikitext),
    max(df_hqq_mark$ppl_c4)
  )
  max_ppl <- ifelse(max_ppl > max_hqq_ppl, max_ppl, max_hqq_ppl)
  min_ppl <- floor(min_ppl * 10) / 10
  max_ppl <- ceiling(max_ppl * 10) / 10
  step <- round((max_ppl - min_ppl) / 15, digits = 2)
  legend_title <- "Method:"
  if (type == "sbab") {
    legend_title <- "SensiBoost Ablation:"
  } else if (type == "kbab") {
    legend_title <- "KurtBoost Ablation:"
  }

  # aes(x = increment, y = ppl, shape = method, color = method)
  plt <- ggplot(df_disp, ass) +
    scale_x_continuous(
      limits = c(0, 8.0),
      breaks = seq(0, 8.0, 0.50)
    ) +
    scale_y_continuous(
      limits = c(min_ppl, max_ppl),
      breaks = seq(min_ppl, max_ppl, step)
    ) +
    geom_point(size = 2) +
    geom_hline(
      yintercept = baseline_451,
      linetype = "dashed",
      linewidth = 0.1
    ) +
    annotate(
      "text",
      x = 7.5, y = baseline_451 - 0.01, size = 3, label = "HQQ b4g32"
    ) +
    geom_hline(
      yintercept = baseline_425,
      linetype = "dashed",
      linewidth = 0.1
    ) +
    annotate(
      "text",
      x = 7.5, y = baseline_425 + 0.01, size = 3, label = "HQQ b4g64"
    ) +
    geom_hline(
      yintercept = baseline_413,
      linetype = "dashed",
      linewidth = 0.1
    ) +
    annotate(
      "text",
      x = 7.5, y = baseline_413 + 0.01, size = 3, label = "HQQ b4g128"
    ) +
    geom_hline(
      yintercept = baseline_351,
      linetype = "dashed",
      linewidth = 0.1
    ) +
    annotate(
      "text",
      x = 7.5, y = baseline_351 + 0.01, size = 3, label = "HQQ b3g32"
    ) +
    geom_hline(
      yintercept = baseline_325,
      linetype = "dashed",
      linewidth = 0.1
    ) +
    annotate(
      "text",
      x = 7.5, y = baseline_325 + 0.01, size = 3, label = "HQQ b3g64"
    ) +
    geom_hline(
      yintercept = baseline_313,
      linetype = "dashed",
      linewidth = 0.1
    ) +
    annotate(
      "text",
      x = 7.5, y = baseline_313 - 0.01, size = 3, label = "HQQ b3g128"
    ) +
    geom_magnify(from = mag_from, to = mag_to, axes = "xy") +
    labs(x = "% Memory Increment", y = "Perplexity") +
    theme(
      strip.background = element_rect(
        color = "darkgray", fill = "white", linewidth = 1.0, linetype = "solid"
      ),
      strip.text.x = element_text(face = "bold", size = 12),
      strip.text.y = element_text(face = "bold", size = 12),
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 12),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      legend.position = "bottom"
    ) +
    guides(
      shape = guide_legend(title = legend_title),
      color = guide_legend(title = legend_title)
    ) +
    facet_grid(dataset ~ model, scales = "free")

  ggsave(
    paste0("pdfs/ppl-", type, "-", model_id, "-", the_dataset, ".pdf"),
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



if (type == "sensi-vs-kurt") {
  plot_func <- plot_sensi_vs_kurt
} else if (type == "sensi-vs-ablation") {
  plot_func <- plot_sensi_vs_ablation
} else if (type == "kurt-vs-ablation") {
  plot_func <- plot_kurt_vs_ablation
}

tup <- load_ppl_mem_inc(allot_cfg_csv_fp, combined_csv_fp)
df_ppl_mem_inc <- tup[[1]]
df_hqq <- tup[[2]]

write.xlsx(df_ppl_mem_inc, "df_ppl_mem_inc.xlsx", asTable = TRUE)
models <- unique(df_ppl_mem_inc$model)
# models <- c("Llama-2-13B")
for (model_id in models) {
  if (model_id == "Llama-2-7B") {
    from_wk <- c(2.5, 3.2, 5.25, 5.38)
    to_wk <- c(5, 7, 5.5, 6.0)
    from_c4 <- c(2.5, 3.2, 7.04, 7.18)
    to_c4 <- c(4, 6, 7.4, 8.08)
    low_bound_scale <- 1.0
  } else if (model_id == "Llama-2-13B") {
    from_wk <- c(1.9, 2.5, 4.66, 4.75)
    to_wk <- c(4.5, 6.5, 4.85, 5.15)
    from_c4 <- c(1.9, 2.5, 6.5, 6.6)
    to_c4 <- c(4, 6.0, 6.7, 7.00)
    low_bound_scale <- 0.99
  } else if (model_id == "Llama-3-8B") {
    from_wk <- c(2.5, 3.2, 6.02, 6.42)
    to_wk <- c(4.0, 6.0, 6.80, 8.65)
    from_c4 <- c(2.5, 3.2, 9.31, 10.04)
    to_c4 <- c(4, 6, 11, 14)
    low_bound_scale <- 0.98
  }
  plot_func(
    model_id,
    "WikiText2",
    df_ppl_mem_inc,
    df_hqq,
    from_wk,
    to_wk,
    low_bound_scale
  )
  plot_func(
    model_id,
    "C4",
    df_ppl_mem_inc,
    df_hqq,
    from_c4,
    to_c4,
    low_bound_scale
  )
}
