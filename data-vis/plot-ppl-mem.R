#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggbreak)
library(readr)
library(patchwork)
library(optparse)
library(ggmagnify)

mxq_attempt <- "MXQ2"
# Plot memory drop vs PPL loss ---------------------------------
plot_ppl <- function(df_disp) {
  guideline_color <- "coral4"
  df_wikitxt <- df_disp |>
    filter(
      dataset == "WikiText"
    )
  min_ppl_wt <- min(df_wikitxt$ppl)
  min_bpp_wt <- min(df_wikitxt$bpp)
  df_c4 <- df_disp |>
    filter(
      dataset == "C4"
    )
  min_ppl_c4 <- min(df_c4$ppl)
  min_bpp_c4 <- min(df_c4$bpp)
  ppl_low_bound <- floor(min_ppl_wt * 10) / 10
  plt <- ggplot(
    subset(df_disp, algo != "MXQ"),
    aes(x = bpp, y = ppl),
  ) +
    geom_point(
      data = subset(df_disp, algo == "MXQ"),
      size = 0.6,
      aes(shape = algo, color = dataset, y = ppl)
    ) +
    geom_point(size = 1.5, aes(shape = algo, color = dataset, y = ppl)) +
    geom_hline(
      yintercept = min_ppl_wt * 1.02,
      linetype = "dashed",
      size = 0.1,
      color = guideline_color
    ) +
    geom_hline(
      yintercept = min_ppl_wt * 1.01,
      linetype = "dashed",
      size = 0.1,
      color = guideline_color
    ) +
    geom_hline(
      yintercept = min_ppl_wt,
      linetype = "dashed",
      size = 0.1,
      color = guideline_color
    ) +
    geom_hline(
      yintercept = min_ppl_c4 * 1.02,
      linetype = "dashed",
      size = 0.1,
      color = guideline_color
    ) +
    geom_hline(
      yintercept = min_ppl_c4 * 1.01,
      linetype = "dashed",
      size = 0.1,
      color = guideline_color
    ) +
    geom_hline(
      yintercept = min_ppl_c4,
      linetype = "dashed",
      size = 0.1,
      color = guideline_color
    ) +
    geom_magnify(
      from = c(4.10, 4.55, 4.65, 4.75),
      to = c(3.70, 5.20, 5.20, 6.20),
      colour = "#fc8d62",
      linewidth = 0.3,
      axes = "xy"
    ) +
    geom_magnify(
      from = c(4.10, 4.55, 6.47, 6.62),
      to = c(3.80, 5.30, 6.85, 7.80),
      colour = "#fc8d62",
      linewidth = 0.3,
      axes = "xy"
    ) +
    annotate("text", x = 15.8, y = min_ppl_wt * 1.00, label = "FP16") +
    annotate("text", x = 15.8, y = min_ppl_c4 * 1.00, label = "FP16") +
    scale_x_break(c(5.5, 15.6)) +
    scale_x_continuous(
      limits = c(3.0, 16.2),
      breaks = seq(3.0, 16.2, 0.25),
      sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memery Reduction")
    ) +
    scale_y_continuous(
      limits = c(ppl_low_bound, min_ppl_c4 * 1.20),
      breaks = seq(
        round(ppl_low_bound, digits = 2),
        round(min_ppl_c4 * 1.20, digits = 2),
        0.25
      ),
      sec.axis = sec_axis(
        ~ 100 * (. - min_ppl_c4) / min_ppl_c4,
        name = "% Degradation",
        breaks = seq(-30, 20, 5),
      )
    ) +
    labs(x = "Bit Budget", y = "Perplexity") +
    theme_gray(base_size = 12) +
    guides(
      shape = guide_legend(title = "Method:"),
      color = guide_legend(title = "Dataset:")
    ) +
    theme(
      legend.position = "left"
    ) +
    facet_wrap(~model, scales = "free") +
    scale_color_solarized()

  return(plt)
}

proc_data <- function(df, model_name) {
  df_disp <- df |>
    filter(
      grepl(mdl, model) & bpp >= 2.5
    ) |>
    filter(is.na(attempt) | attempt == mxq_attempt) |>
    pivot_longer(
      cols = c("ppl_wikitext", "ppl_c4"),
      names_to = c(".value", "dataset"),
      names_sep = "_"
    ) |>
    mutate(
      dataset = factor(
        dataset,
        levels = c(
          "wikitext",
          "c4"
        ),
        labels = c(
          "WikiText",
          "C4"
        ),
      )
    )
  return(df_disp)
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--csv_file"),
  type = "character",
  help = "Combined PPL result CSV file",
  metavar = "character"
)
args <- parse_args(parser)

if (is.null(args$csv_file)) {
  csv_fp <- "data/combined.csv"
} else {
  csv_fp <- args$csv_file
}

all_cols <- c(
  "model", "algo", "config", "attempt",
  "bpp", "ppl_wikitext", "ppl_c4"
)
df_all <- read_csv(csv_fp) |>
  select(all_of(all_cols)) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Meta-Llama-3-8B", "Llama-2-13b-hf"),
      labels = c("Llama-2-7B", "Llama-3-8B", "Llama-2-13B")
    ),
    algo = factor(
      algo,
      levels = c("mxq", "fp16", "awq", "gptq", "bnb", "hqq"),
      labels = c("MXQ", "FP16", "AWQ", "GPTQ", "BnB", "HQQ"),
    ),
    attempt = factor(
      attempt,
      levels = c(
        "mxq1",
        "kurt-scaled"
      ),
      labels = c(
        "MXQ1",
        "KURT-SCALED"
      ),
    )
  )

models <- unique(df_all$model)
for (mdl in models) {
  df_disp <- proc_data(df_all, mdl)
  plt <- plot_ppl(df_disp)

  pdf.options(reset = TRUE, onefile = FALSE)
  ggsave(
    paste0("pdfs/ppl-mem-", mdl, ".pdf"),
    plot = plt,
    width = 8,
    height = 5
  )
}

# mdl <- "Llama-2-13B"
# df_disp <- proc_data(df_all, mdl)
# plt <- plot_ppl(df_disp)
#
# pdf.options(reset = TRUE, onefile = FALSE)
# ggsave(
#   paste0("pdfs/ppl-mem-", mdl, ".png"),
#   plot = plt,
#   width = 8,
#   height = 5,
#   dpi = 600
# )
