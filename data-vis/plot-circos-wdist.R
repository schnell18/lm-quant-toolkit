#!/usr/bin/env Rscript

library(tidyverse)
library(readr)
library(optparse)
library(circlize)

plot_wdist_circle <- function(df, model_id, draw_sensitivity = TRUE) {
  circos.clear()
  circos.par("track.height" = 0.3)
  circos.par("gap.degree" = 7)
  circos.initialize(df$module, x = df$layer)
  circos.track(
    ylim = c(0, 2),
    panel.fun = function(x, y) {
      circos.text(
        CELL_META$xcenter,
        CELL_META$cell.ylim[2] - 0.3,
        CELL_META$sector.index,
        facing = "bending.inside",
        niceFacing = TRUE
      )
      if (CELL_META$sector.index == "self_attn.v_proj") {
        circos.text(
          CELL_META$xcenter,
          CELL_META$cell.ylim[2] - 1.3,
          "ABS Value",
          cex = 0.7,
          facing = "bending.inside",
          niceFacing = TRUE
        )
      }
      circos.yaxis(
        "left",
        labels.cex = 0.7,
      )
      circos.axis(
        labels.cex = 0.7,
        major.at = c(0, 5, 10, 15, 20, 25, 30, 35, 40)
      )
      df_mod <- df |>
        filter(module == CELL_META$sector.index)
      df_pct <- df_mod |>
        select(
          c(
            "percentile_0",
            "percentile_99",
            "percentile_999",
            "percentile_9999",
            "percentile_100"
          )
        )
      circos.barplot(
        as.matrix(df_pct),
        df_mod$layer,
        border = "darkgray",
        col = 2:6
      )
    }
  )

  if (draw_sensitivity) {
    circos.track(
      df$module,
      x = df$layer,
      y = df$sensitivity,
      track.height = 0.15,
      panel.fun = function(x, y) {
        df_mod <- df |>
          filter(module == CELL_META$sector.index)

        if (CELL_META$sector.index == "self_attn.q_proj") {
          circos.text(
            CELL_META$xcenter,
            CELL_META$cell.ylim[1] + 1.1,
            "Sensitivity",
            cex = 0.7,
            facing = "bending.inside",
            niceFacing = TRUE
          )
        }
        circos.lines(df_mod$layer, df_mod$sensitivity, col = "brown")
      }
    )
  }


  circos.track(
    df$module,
    x = df$layer,
    y = df$kurtosis,
    track.height = 0.15,
    panel.fun = function(x, y) {
      if (CELL_META$sector.index == "self_attn.o_proj") {
        circos.text(
          CELL_META$xcenter,
          CELL_META$ycenter,
          "Kurtosis",
          cex = 0.7,
          facing = "bending.inside",
          niceFacing = TRUE
        )
      }
      circos.lines(x, y, col = "blue")
    }
  )
  text(0, 0, model_id, cex = 2, col = "darkblue")
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
parser <- add_option(
  parser, c("-s", "--sensi_data_dir"),
  type = "character",
  help = "The data directory of sensitivity data",
  metavar = "character"
)

args <- parse_args(parser)

if (is.null(args$model_id)) {
  model_id <- "Meta-Llama-3-8B"
} else {
  model_id <- args$model_id
}
if (is.null(args$wdist_dir)) {
  wdist_dir <- "data/wdist"
} else {
  wdist_dir <- args$wdist_dir
}
if (is.null(args$sensi_data_dir)) {
  sensi_data_dir <- "../src/data"
} else {
  sensi_data_dir <- args$sensi_data_dir
}

list1_dfs <- list()
list2_dfs <- list()
for (model_id in c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B")) {
  df_one <- read_csv(paste0(wdist_dir, "/wdist-", model_id, ".csv"))
  df_one$model <- model_id
  list1_dfs <- append(list1_dfs, list(df_one))
  df_two <- read_csv(paste0(sensi_data_dir, "/fnorm-", model_id, ".csv"))
  df_two$model <- model_id
  list2_dfs <- append(list2_dfs, list(df_two))
}
df_wdist <- bind_rows(list1_dfs)
df_sensi <- bind_rows(list2_dfs) |>
  filter(nbit1 == 4 & gsize1 == 64) |>
  group_by(model, module, layer) |>
  summarise(
    sensitivity = min(sensitivity)
  ) |>
  mutate(
    sensitivity = log10(sensitivity)
  ) |>
  ungroup()

percentiles <- c(
  "percentile_0",
  "percentile_99",
  "percentile_999",
  "percentile_9999",
  "percentile_100"
)

df_module_param_count <- df_wdist |>
  select(
    model, module, param_count
  ) |>
  group_by(model, module) |>
  summarise(
    param_count = sum(param_count)
  ) |>
  mutate(
    mod_disp = paste0(module, "(", formatC(param_count, big.mark = ","), ")")
  )

df_wdist <- df_wdist |>
  filter(module != "input_layernorm" & module != "post_attention_layernorm") |>
  left_join(df_sensi, by = c("model", "module", "layer")) |>
  left_join(df_module_param_count, by = c("model", "module")) |>
  select(
    c(
      "model",
      "module",
      "layer",
      all_of(percentiles),
      "kurtosis",
      "sensitivity"
    )
  ) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
      labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
    ),
  )

models <- unique(df_wdist$model)
model_cnt <- length(models)
# layout(matrix(1:model_cnt, 1, model_cnt))
for (model_id in models) {
  pdf(
    paste0("pdfs/circos-", model_id, ".pdf"),
    width = 8,
    height = 8
  )
  df_disp <- df_wdist |> filter(model == model_id)
  plot_wdist_circle(df_disp, model_id, draw_sensitivity = FALSE)
  circos.clear()
  dev.off()
}
