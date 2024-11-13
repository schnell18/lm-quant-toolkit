#!/usr/bin/env Rscript

library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(ggthemes)
library(ggplot2)
library(patchwork)
library(optparse)
library(RColorBrewer)

plot_allot_pair <- function(df_lower, df_upper) {
  mem_plot <- ggplot(
    df_upper, aes(x = layer, y = memmb, color = "blue")
  ) +
    geom_line() +
    theme_gray(base_size = 14) +
    labs(x = "", y = "Memory(MiB)") +
    theme(
      legend.position = "none",
      axis.title.x = element_blank(),
      axis.text.x = element_blank()
    ) +
    scale_color_solarized()

  # Bar plot (on bottom)
  if (is.na(df_lower$attempt[1])) {
    module_disp <- paste0(df_lower$module[1], "(HQQ)")
  } else {
    module_disp <- paste0(df_lower$module[1], "(", df_lower$attempt[1], ")")
  }
  bar_plot <- ggplot(
    df_lower, aes(x = layer, y = fnorm, fill = cfg, color = cfg)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    geom_text(
      aes(x = layer, label = cfg),
      angle = 90,
      vjust = 0.20,
      position = position_stack(vjust = 0.5),
      colour = "black",
      size = 2
    ) +
    theme_gray(base_size = 14) +
    labs(x = module_disp, y = "FNorm") +
    theme(legend.position = "none") +
    col_scale

  # Combine the line and bar plot vertically
  combined_plot <- mem_plot / bar_plot + plot_layout(heights = c(1, 2))
  return(combined_plot)
}

weight_grid <- function(df_disp, mod, attmpt) {
  if (is.null(attmpt)) {
    df_plot <- df_disp |> filter(module == mod & is.na(attempt))
  } else {
    df_plot <- df_disp |> filter(module == mod & attempt == attmpt)
  }
  plot_allot_pair(df_plot, df_plot)
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
  baseline_data_dir <- "data"
} else {
  baseline_data_dir <- args$baseline_data_dir
}

if (is.null(args$quant_cfg_allot_file)) {
  quant_cfg_allot_file <- "data/quant-cfg-allocation.csv"
} else {
  quant_cfg_allot_file <- args$quant_cfg_allot_file
}

fnorm_dir <- path.expand(paste0(baseline_data_dir, "/", "fnorm"))
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = "fnorm-.*\\.csv$",
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- ldply(fnorm_fps, read.csv, stringsAsFactors = FALSE, .id = "model")

k_cols <- c("model", "module", "layer", "cfg", "fnorm", "memmb", "params")
df_fnorm <- df_fnorm |>
  filter(
    model == model_id
  ) |>
  mutate(
    cfg = paste0("b", nbit1, "g", gsize1)
  ) |>
  select(all_of(k_cols)) |>
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

df_cfgs <- read_csv("data/quant-cfg-allocation.csv")

df_cfg_1 <- df_cfgs |>
  filter(bit_budget == budget & model == model_id) |>
  mutate(
    cfg = paste0("b", b1, "g", g1)
  ) |>
  select(-c("b1", "g1", "b2", "g2", "bit_budget", "memmb")) |>
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

by <- join_by(model == model, module == module, layer == layer, cfg == cfg)
df_disp <- df_cfg_1 |>
  left_join(df_fnorm, by)

my_colors <- brewer.pal(12, "Paired")
names(my_colors) <- levels(df_disp$cfg)
col_scale <- scale_fill_manual(name = "cfg", values = my_colors)

p11 <- weight_grid(df_disp, "mlp.down_proj", NULL)
p12 <- weight_grid(df_disp, "mlp.down_proj", "mxq1")
p13 <- weight_grid(df_disp, "mlp.down_proj", "kurt-scaled")

p21 <- weight_grid(df_disp, "mlp.gate_proj", NULL)
p22 <- weight_grid(df_disp, "mlp.gate_proj", "mxq1")
p23 <- weight_grid(df_disp, "mlp.gate_proj", "kurt-scaled")

p31 <- weight_grid(df_disp, "mlp.up_proj", NULL)
p32 <- weight_grid(df_disp, "mlp.up_proj", "mxq1")
p33 <- weight_grid(df_disp, "mlp.up_proj", "kurt-scaled")

final_plot <- (p11 | p21 | p31) / (p12 | p22 | p32) / (p13 | p23 | p33)
final_plot
ggsave(
  paste0("pdfs/", model_id, "-", budget, "-mlp-allot.pdf"),
  width = 16, height = 9
)

p61 <- weight_grid(df_disp, "self_attn.k_proj", NULL)
p62 <- weight_grid(df_disp, "self_attn.k_proj", "mxq1")
p63 <- weight_grid(df_disp, "self_attn.k_proj", "kurt-scaled")

p71 <- weight_grid(df_disp, "self_attn.o_proj", NULL)
p72 <- weight_grid(df_disp, "self_attn.o_proj", "mxq1")
p73 <- weight_grid(df_disp, "self_attn.o_proj", "kurt-scaled")

p81 <- weight_grid(df_disp, "self_attn.q_proj", NULL)
p82 <- weight_grid(df_disp, "self_attn.q_proj", "mxq1")
p83 <- weight_grid(df_disp, "self_attn.q_proj", "kurt-scaled")

p91 <- weight_grid(df_disp, "self_attn.v_proj", NULL)
p92 <- weight_grid(df_disp, "self_attn.v_proj", "mxq1")
p93 <- weight_grid(df_disp, "self_attn.v_proj", "kurt-scaled")

final_plot2 <- (p61 | p71 | p81 | p91) / (p62 | p72 | p82 | p92) / (p63 | p73 | p83 | p93)
final_plot2
ggsave(
  paste0("pdfs/", model_id, "-", budget, "-attn-allot.pdf"),
  width = 16, height = 9
)
