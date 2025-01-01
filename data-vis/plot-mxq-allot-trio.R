#!/usr/bin/env Rscript

library(tidyverse)
library(dplyr)
library(readr)
library(ggthemes)
library(ggplot2)
library(patchwork)
library(optparse)
library(RColorBrewer)

plot_allot_pair <- function(df_lower, df_upper, disp_fnorm = TRUE) {
  ylabel <- ifelse(disp_fnorm == TRUE, "FNorm", "Sensitivity")
  mem_plot <- ggplot(
    df_upper, aes(x = layer, y = memmb, color = "blue")
  ) +
    geom_line() +
    theme_gray(base_size = 12) +
    labs(x = "", y = "Memory(MiB)") +
    theme(
      legend.position = "none",
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      axis.ticks = element_blank()
    ) +
    scale_color_solarized()

  metric_cutoff <- 1.0
  if (is.na(df_lower$attempt[1])) {
    module_disp <- paste0("HQQ - ", df_lower$module[1])
  } else {
    module_disp <- paste0(
      R.utils::toCamelCase(df_lower$attempt[1], split = "-", capitalize = TRUE),
      " - ",
      df_lower$module[1]
    )
  }
  bar_plot <- ggplot(
    df_lower, aes(x = layer, y = metric, fill = cfg, color = cfg)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    geom_text(
      data = subset(df_lower, metric <= metric_cutoff),
      aes(x = layer, label = cfg),
      colour = "black",
      angle = 90,
      hjust = -0.20,
      size = 2
    ) +
    geom_text(
      data = subset(df_lower, metric > metric_cutoff),
      aes(x = layer, label = cfg),
      colour = "black",
      angle = 90,
      vjust = 0.20,
      position = position_stack(vjust = 0.5),
      size = 2
    ) +
    theme_gray(base_size = 12) +
    labs(x = module_disp, y = "FNorm") +
    theme(legend.position = "none") +
    col_scale

  # Combine the line and bar plot vertically
  combined_plot <- mem_plot / bar_plot + plot_layout(heights = c(1, 2))
  return(combined_plot)
}

weight_grid <- function(df_disp, mod, attmpt, bar_value_fnorm) {
  if (is.null(attmpt)) {
    df_plot <- df_disp |> filter(module == mod & is.na(attempt))
  } else {
    df_plot <- df_disp |> filter(module == mod & attempt == attmpt)
  }
  plot_allot_pair(df_plot, df_plot, bar_value_fnorm)
}

strip_name <- function(name) {
  start <- nchar("fnorm-") + 1
  stop <- nchar(name) - 4
  return(substr(name, start, stop))
}

match_hqq_base_config <- function(budget) {
  buckets <- c(
    2.13, 2.25, 2.51, 3.13, 3.25, 3.51, 4.13, 4.25, 4.51, 8.13, 8.25, 8.51
  )
  idx <- which.min(abs(buckets - budget))
  return(buckets[[idx]])
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-m", "--model"),
  type = "character",
  help = "Model ID",
  metavar = "character"
)
parser <- add_option(
  parser, c("-b", "--budget"),
  type = "double",
  help = "Bit Budget",
  metavar = "double"
)
parser <- add_option(
  parser, c("-d", "--baseline_data_dir"),
  type = "character",
  help = "Data directory of baseline results",
  metavar = "character"
)
parser <- add_option(
  parser, c("-q", "--quant_cfg_allot_file"),
  type = "character",
  help = "The combined quant config allocation csv file",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt1"),
  type = "character",
  help = "The first attempt to plot",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt2"),
  type = "character",
  help = "The second attempt to plot",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt3"),
  type = "character",
  help = "The third attempt to plot",
  metavar = "character"
)
parser <- add_option(
  parser, c("--fnorm"),
  action = "store_true",
  default = TRUE,
  type = "logical",
  help = "Display FNorm value in the bar chart",
  metavar = "logical"
)


args <- parse_args(parser)

if (is.null(args$model)) {
  model_id <- "Llama-2-7b-hf"
} else {
  model_id <- args$model
}

if (is.null(args$budget)) {
  budget <- 4.51
} else {
  budget <- args$budget
}

if (is.null(args$baseline_data_dir)) {
  baseline_data_dir <- "../src/data"
} else {
  baseline_data_dir <- args$baseline_data_dir
}

if (is.null(args$quant_cfg_allot_file)) {
  quant_cfg_allot_file <- "data/quant-cfg-allocation.csv"
} else {
  quant_cfg_allot_file <- args$quant_cfg_allot_file
}

if (is.null(args$output_dir)) {
  output_dir <- "pdfs/allot"
} else {
  output_dir <- args$output_dir
}

if (is.null(args$attempt1)) {
  attempt1 <- "mxq1"
} else {
  attempt1 <- args$attempt1
}

if (is.null(args$attempt2)) {
  attempt2 <- "kurt-scaled"
} else {
  attempt2 <- args$attempt2
}

if (is.null(args$attempt3)) {
  attempt3 <- "kurt-scaled"
} else {
  attempt3 <- args$attempt3
}

if (is.null(args$fnorm)) {
  bar_value_fnorm <- FALSE
} else {
  bar_value_fnorm <- args$fnorm
}

fnorm_dir <- path.expand(baseline_data_dir)
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = paste0("fnorm-", model_id, "\\.csv$"),
  # pattern = "fnorm-.*\\.csv$",
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- plyr::ldply(
  fnorm_fps,
  read.csv,
  stringsAsFactors = FALSE,
  .id = "model"
)

k_cols <- c(
  "model", "module", "layer", "cfg", "fnorm", "sensitivity", "memmb", "params"
)
df_fnorm <- df_fnorm |>
  filter(
    model == model_id
  ) |>
  dplyr::mutate(
    cfg = paste0("b", nbit1, "g", gsize1)
  ) |>
  select(all_of(k_cols)) |>
  dplyr::mutate(
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

if (bar_value_fnorm) {
  df_fnorm <- df_fnorm |>
    dplyr::rename(metric = fnorm) |>
    select(-c("sensitivity"))
} else {
  df_fnorm <- df_fnorm |>
    dplyr::rename(metric = sensitivity) |>
    select(-c("fnorm"))
}

rounded_budget <- match_hqq_base_config(budget)
df_cfgs <- read_csv("data/quant-cfg-allocation.csv")

df_cfg_1 <- df_cfgs |>
  filter(
    (bit_budget == budget | bit_budget == rounded_budget) & model == model_id
  ) |>
  dplyr::mutate(
    cfg = paste0("b", b1, "g", g1)
  ) |>
  select(-c("b1", "g1", "b2", "g2", "bit_budget", "memmb")) |>
  dplyr::mutate(
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

by <- join_by(model == model, module == module, layer == layer, cfg == cfg)
df_disp <- df_cfg_1 |>
  left_join(df_fnorm, by)

my_colors <- brewer.pal(12, "Paired")
names(my_colors) <- levels(df_disp$cfg)
col_scale <- scale_fill_manual(name = "cfg", values = my_colors)

p11 <- weight_grid(df_disp, "mlp.down_proj", attempt1, bar_value_fnorm)
p12 <- weight_grid(df_disp, "mlp.down_proj", attempt2, bar_value_fnorm)
p13 <- weight_grid(df_disp, "mlp.down_proj", attempt3, bar_value_fnorm)

p21 <- weight_grid(df_disp, "mlp.gate_proj", attempt1, bar_value_fnorm)
p22 <- weight_grid(df_disp, "mlp.gate_proj", attempt2, bar_value_fnorm)
p23 <- weight_grid(df_disp, "mlp.gate_proj", attempt3, bar_value_fnorm)

p31 <- weight_grid(df_disp, "mlp.up_proj", attempt1, bar_value_fnorm)
p32 <- weight_grid(df_disp, "mlp.up_proj", attempt2, bar_value_fnorm)
p33 <- weight_grid(df_disp, "mlp.up_proj", attempt3, bar_value_fnorm)

final_plot <- (p11 | p21 | p31) / (p12 | p22 | p32) / (p13 | p23 | p33)
final_plot
ggsave(
  paste0(output_dir, "/", model_id, "-", budget, "-mlp-allot.pdf"),
  create.dir = TRUE,
  width = 16, height = 9
)

p61 <- weight_grid(df_disp, "self_attn.k_proj", attempt1, bar_value_fnorm)
p62 <- weight_grid(df_disp, "self_attn.k_proj", attempt2, bar_value_fnorm)
p63 <- weight_grid(df_disp, "self_attn.k_proj", attempt3, bar_value_fnorm)

p71 <- weight_grid(df_disp, "self_attn.o_proj", attempt1, bar_value_fnorm)
p72 <- weight_grid(df_disp, "self_attn.o_proj", attempt2, bar_value_fnorm)
p73 <- weight_grid(df_disp, "self_attn.o_proj", attempt3, bar_value_fnorm)

p81 <- weight_grid(df_disp, "self_attn.q_proj", attempt1, bar_value_fnorm)
p82 <- weight_grid(df_disp, "self_attn.q_proj", attempt2, bar_value_fnorm)
p83 <- weight_grid(df_disp, "self_attn.q_proj", attempt3, bar_value_fnorm)

p91 <- weight_grid(df_disp, "self_attn.v_proj", attempt1, bar_value_fnorm)
p92 <- weight_grid(df_disp, "self_attn.v_proj", attempt2, bar_value_fnorm)
p93 <- weight_grid(df_disp, "self_attn.v_proj", attempt3, bar_value_fnorm)

final_plot2 <- (p61 | p71 | p81 | p91) / (p62 | p72 | p82 | p92) / (p63 | p73 | p83 | p93)
final_plot2
ggsave(
  paste0(output_dir, "/", model_id, "-", budget, "-attn-allot.pdf"),
  create.dir = TRUE,
  width = 16, height = 9
)
