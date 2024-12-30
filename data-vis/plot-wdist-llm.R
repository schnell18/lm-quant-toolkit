#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggplot2)
library(patchwork)
library(readr)
library(optparse)

weight_grid <- function(
    df_wdist, df_kurtosis, mod, show_legend = FALSE) {
  df_mod_wdist <- df_wdist |> filter(module == mod)
  df_mod_kurt <- df_kurtosis |> filter(module == mod)
  # Line plot (on top)
  line_plot <- ggplot(df_mod_kurt, aes(x = layer, y = kurtosis)) +
    geom_line(color = "blue") +
    theme_gray(base_size = 14) +
    theme_minimal() +
    labs(y = "Kurtosis") +
    theme(
      axis.title.y = element_text(size = 12),
      axis.title.x = element_blank(),
      axis.text.x = element_blank()
    )

  # Bar plot (on bottom)
  module_disp <- df_mod_wdist$mod_disp[1]
  bar_plot <- ggplot(
    df_mod_wdist,
    aes(x = layer, y = abs_val, fill = nth_percentile)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    theme_gray(base_size = 14) +
    labs(
      x = module_disp, y = "Absolute Value", fill = "Percentile"
    )

  if (show_legend) {
    bar_plot <- bar_plot +
      theme(
        axis.title.y = element_text(size = 12),
        legend.position = "bottom",
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 12)
      ) +
      guides(fill = guide_legend(nrow = 1)) +
      scale_color_solarized()
  } else {
    bar_plot <- bar_plot +
      theme(legend.position = "none") +
      scale_color_solarized()
  }

  # Combine the line and bar plot vertically
  combined_plot <- line_plot / bar_plot + plot_layout(heights = c(1, 3))
  return(combined_plot)
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-m", "--model_id"),
  type = "character",
  help = "The short HF model id without the organization prefix",
  metavar = "character"
)
parser <- add_option(
  parser, c("-d", "--wdist_dir"),
  type = "character",
  help = "The data directory of weight distribution data",
  metavar = "character"
)

args <- parse_args(parser)

if (is.null(args$model_id)) {
  model_id <- "Llama-2-13b-hf"
} else {
  model_id <- args$model_id
}
if (is.null(args$wdist_dir)) {
  wdist_dir <- "data/wdist"
} else {
  wdist_dir <- args$wdist_dir
}

df_wdist <- read_csv(paste0(wdist_dir, "/wdist-", model_id, ".csv"))
percentiles <- c("0", "99", "99.9", "99.99", "100")
df_module_param_count <- df_wdist |>
  select(
    module, param_count
  ) |>
  group_by(module) |>
  summarise(
    param_count = sum(param_count)
  ) |>
  mutate(
    mod_disp = paste0(module, "(", formatC(param_count, big.mark = ","), ")")
  )

df_kurtosis <- df_wdist |>
  select(c("module", "layer", "kurtosis"))

df_wdist <- df_wdist |>
  mutate(
    `0` = percentile_0,
    `99` = percentile_99 - percentile_0,
    `99.9` = percentile_999 - percentile_99,
    `99.99` = percentile_9999 - percentile_999,
    `100` = percentile_100 - percentile_9999,
  ) |>
  select(c("module", "layer", "kurtosis", all_of(percentiles))) |>
  pivot_longer(
    cols = all_of(percentiles),
    names_to = "nth_percentile",
    names_transform = list(nth_percentile = as.numeric),
    values_to = "abs_val"
  ) |>
  mutate(
    nth_percentile = factor(nth_percentile, levels = rev(percentiles))
  ) |>
  left_join(df_module_param_count, by = c("module"))

p1 <- weight_grid(df_wdist, df_kurtosis, "input_layernorm")
p2 <- weight_grid(df_wdist, df_kurtosis, "mlp.down_proj")
p3 <- weight_grid(df_wdist, df_kurtosis, "mlp.gate_proj")
p4 <- weight_grid(df_wdist, df_kurtosis, "mlp.up_proj")
p5 <- weight_grid(df_wdist, df_kurtosis, "post_attention_layernorm")
p6 <- weight_grid(df_wdist, df_kurtosis, "self_attn.k_proj")
p7 <- weight_grid(df_wdist, df_kurtosis, "self_attn.o_proj")
p8 <- weight_grid(df_wdist, df_kurtosis, "self_attn.q_proj", TRUE)
p9 <- weight_grid(df_wdist, df_kurtosis, "self_attn.v_proj")

# Create a 3x3 grid of combined plots
final_plot <- (p1 | p2 | p3) / (p4 | p5 | p6) / (p7 | p8 | p9)
final_plot
ggsave(
  paste0("pdfs/", model_id, "-wdist-kurtosis.pdf"),
  width = 16, height = 9
)
