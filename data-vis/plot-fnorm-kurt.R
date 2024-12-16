#!/usr/bin/env Rscript

library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(ggthemes)
library(ggplot2)
library(patchwork)
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
      axis.title.x = element_blank(),
      axis.text.x = element_blank()
    )

  # Bar plot (on bottom)
  module_disp <- df_mod_wdist$module[1]
  bar_plot <- ggplot(
    df_mod_wdist, aes(x = layer, y = fnorm, fill = cfg)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    theme_gray(base_size = 14) +
    labs(
      x = module_disp, y = "FNorm", fill = "cfg"
    )

  if (show_legend) {
    bar_plot <- bar_plot +
      theme(
        legend.position = "bottom",
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14)
      ) +
      guides(fill = guide_legend(nrow = 3)) +
      # coord_flip() +
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

strip_name <- function(name) {
  start <- nchar("fnorm-") + 1
  stop <- nchar(name) - 4
  return(substr(name, start, stop))
}


parser <- OptionParser()
parser <- add_option(
  parser, c("-m", "--model"),
  type = "character",
  help = "Model ID",
  metavar = "character"
)
args <- parse_args(parser)

if (is.null(args$model)) {
  model_id <- "Llama-2-7b-hf"
} else {
  model_id <- args$model
}

fnorm_dir <- path.expand("../src/data")
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = "fnorm-.*\\.csv$",
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- ldply(fnorm_fps, read.csv, stringsAsFactors = FALSE, .id = "model")

k_cols <- c("module", "layer", "cfg", "fnorm", "kurtosis")
df_wdist <- df_fnorm |>
  filter(
    model == model_id
  ) |>
  mutate(
    cfg = paste0("b", nbit1, "g", gsize1)
  ) |>
  select(all_of(k_cols)) |>
  pivot_wider(
    names_from = "cfg",
    values_from = "fnorm"
  ) |>
  mutate(
    b2g128_fnorm = b2g128 - b2g64,
    b2g64_fnorm = b2g64 - b2g32,
    b2g32_fnorm = b2g32 - b3g128,
    b3g128_fnorm = b3g128 - b3g64,
    b3g64_fnorm = b3g64 - b3g32,
    b3g32_fnorm = b3g32 - b4g128,
    b4g128_fnorm = b4g128 - b4g64,
    b4g64_fnorm = b4g64 - b4g32,
    b4g32_fnorm = b4g32 - b8g128,
    b8g128_fnorm = b8g128 - b8g64,
    b8g64_fnorm = b8g64 - b8g32,
    b8g32_fnorm = b8g32
  ) |>
  select(
    c(
      "module",
      "layer",
      "kurtosis",
      "b2g128_fnorm",
      "b2g64_fnorm",
      "b2g32_fnorm",
      "b3g128_fnorm",
      "b3g64_fnorm",
      "b3g32_fnorm",
      "b4g128_fnorm",
      "b4g64_fnorm",
      "b4g32_fnorm",
      "b8g128_fnorm",
      "b8g64_fnorm",
      "b8g32_fnorm",
    )
  ) |>
  pivot_longer(
    cols = ends_with(c("_fnorm")),
    names_to = c("cfg", ".value"),
    names_sep = "_"
  ) |>
  mutate(
    cfg = factor(
      cfg,
      levels = c(
        "b2g128", "b2g64", "b2g32",
        "b3g128", "b3g64", "b3g32",
        "b4g128", "b4g64", "b4g32",
        "b8g128", "b8g64", "b8g32"
      )
    )
  )

df_kurtosis <- df_wdist |>
  group_by(module, layer) |>
  dplyr::summarise(
    kurtosis = max(kurtosis)
  ) |>
  ungroup()

p2 <- weight_grid(df_wdist, df_kurtosis, "mlp.down_proj")
p3 <- weight_grid(df_wdist, df_kurtosis, "mlp.gate_proj", TRUE)
p4 <- weight_grid(df_wdist, df_kurtosis, "mlp.up_proj")
p6 <- weight_grid(df_wdist, df_kurtosis, "self_attn.k_proj")
p7 <- weight_grid(df_wdist, df_kurtosis, "self_attn.o_proj")
p8 <- weight_grid(df_wdist, df_kurtosis, "self_attn.q_proj")
p9 <- weight_grid(df_wdist, df_kurtosis, "self_attn.v_proj")

# Create a 3x3 grid of combined plots
final_plot <- (p6 | p7 | p8 | p9) / (p2 | p3 | p4)
final_plot
ggsave(
  paste0("pdfs/", model_id, "-fnorm-kurtosis.pdf"),
  width = 16, height = 9
)
