library(tidyverse)
library(readr)
library(ggthemes)
library(ggplot2)
library(patchwork)

weight_grid <- function(
    df_wdist, df_kurtosis, mod, show_legend = FALSE, show_cfg = TRUE) {
  df_mod_wdist <- df_wdist |> filter(module == mod)
  df_mod_kurt <- df_kurtosis |> filter(module == mod)
  # Line plot (on top)
  line_plot <- ggplot(df_mod_kurt, aes(x = layer, y = kurtosis)) +
    geom_line(color = "blue") +
    theme_gray(base_size = 14) +
    theme_minimal() +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_blank()
    )

  # Bar plot (on bottom)
  module_disp <- df_mod_wdist$mod_disp[1]
  bar_plot <- ggplot(
    df_mod_wdist, aes(x = layer, y = abs_val, fill = nth_percentile)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    theme_gray(base_size = 14) +
    labs(
      x = module_disp, y = "Absolute Value", fill = "nth percentile"
    )
  if (show_cfg) {
    bar_plot <- bar_plot +
      geom_text(
        data = subset(df_mod_wdist, nth_percentile == 100),
        aes(x = layer, label = quant_cfg),
        angle = 90,
        vjust = 0.20,
        position = position_stack(vjust = 0.5),
        colour = "white",
        size = 2
      )
  }

  if (show_legend) {
    bar_plot <- bar_plot +
      theme(
        legend.position = "bottom",
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16)
      ) +
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

weight_grid_only <- function(
    df_wdist, df_kurtosis, mod, show_legend = FALSE) {
  return(weight_grid(df_wdist, df_kurtosis, mod, show_legend, show_cfg = FALSE))
}

plot_quant_cfg <- function(model_id, budget, attempt, cfg_csv_fp) {
  df_cfg_all <- read_csv(cfg_csv_fp)
  df_cfg <- df_cfg_all |>
    filter(bit_budget == budget) |>
    mutate(
      quant_cfg = paste0("b", b1, "g", g1)
    ) |>
    select(-c("b1", "g1", "b2", "g2", "bit_budget"))

  percentiles <- c("0", "99", "99.9", "99.99", "100")
  all_cols <- c("module", "layer", percentiles)
  df_wdist <- df_all |>
    mutate(
      `0` = percentile_0,
      `99` = percentile_99 - percentile_0,
      `99.9` = percentile_999 - percentile_99,
      `99.99` = percentile_9999 - percentile_999,
      `100` = percentile_100 - percentile_9999,
    ) |>
    select(all_of(all_cols)) |>
    pivot_longer(
      cols = percentiles,
      names_to = "nth_percentile",
      names_transform = list(nth_percentile = as.numeric),
      values_to = "abs_val"
    ) |>
    mutate(
      nth_percentile = factor(nth_percentile, levels = rev(percentiles))
    ) |>
    left_join(df_module_param_count, by = c("module")) |>
    left_join(df_cfg, by = c("module", "layer"))


  k_cols <- c("module", "layer", "kurtosis")
  df_kurtosis <- df_all |>
    select(all_of(k_cols))

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
  ggsave(
    paste0("pdfs/", model_id, "-mxq-cfgs-from-model-", attempt, ".pdf"),
    width = 16,
    height = 9
  )
  return(final_plot)
}

plot_wdist <- function(model_id, budget, cfg_csv_fp) {
  df_cfg_all <- read_csv(cfg_csv_fp)
  df_cfg <- df_cfg_all |>
    filter(bit_budget == budget) |>
    mutate(
      quant_cfg = paste0("b", b1, "g", g1)
    ) |>
    select(-c("b1", "g1", "b2", "g2", "bit_budget"))

  percentiles <- c("0", "99", "99.9", "99.99", "100")
  all_cols <- c("module", "layer", percentiles)
  df_wdist <- df_all |>
    mutate(
      `0` = percentile_0,
      `99` = percentile_99 - percentile_0,
      `99.9` = percentile_999 - percentile_99,
      `99.99` = percentile_9999 - percentile_999,
      `100` = percentile_100 - percentile_9999,
    ) |>
    select(all_of(all_cols)) |>
    pivot_longer(
      cols = percentiles,
      names_to = "nth_percentile",
      names_transform = list(nth_percentile = as.numeric),
      values_to = "abs_val"
    ) |>
    mutate(
      nth_percentile = factor(nth_percentile, levels = rev(percentiles))
    ) |>
    left_join(df_module_param_count, by = c("module")) |>
    left_join(df_cfg, by = c("module", "layer"))


  k_cols <- c("module", "layer", "kurtosis")
  df_kurtosis <- df_all |>
    select(all_of(k_cols))

  p1 <- weight_grid_only(df_wdist, df_kurtosis, "input_layernorm")
  p2 <- weight_grid_only(df_wdist, df_kurtosis, "mlp.down_proj")
  p3 <- weight_grid_only(df_wdist, df_kurtosis, "mlp.gate_proj")
  p4 <- weight_grid_only(df_wdist, df_kurtosis, "mlp.up_proj")
  p5 <- weight_grid_only(df_wdist, df_kurtosis, "post_attention_layernorm")
  p6 <- weight_grid_only(df_wdist, df_kurtosis, "self_attn.k_proj")
  p7 <- weight_grid_only(df_wdist, df_kurtosis, "self_attn.o_proj")
  p8 <- weight_grid_only(df_wdist, df_kurtosis, "self_attn.q_proj", TRUE)
  p9 <- weight_grid_only(df_wdist, df_kurtosis, "self_attn.v_proj")

  # Create a 3x3 grid of combined plots
  final_plot <- (p1 | p2 | p3) / (p4 | p5 | p6) / (p7 | p8 | p9)
  ggsave(
    paste0("pdfs/", model_id, "-wdist-kurtosis.pdf"),
    width = 16, height = 9
  )
  return(final_plot)
}


model_id <- "Llama-2-13b-hf"
df_all <- read_csv(paste0("data/wdist/wdist-", model_id, ".csv"))
df_module_param_count <- df_all |>
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

budget <- 4.51
attempt <- "MXQ1"
cfg_csv_fp <- "data/llama-mxq-cfgs.csv"
plot_quant_cfg(model_id, budget, attempt, cfg_csv_fp)

attempt <- "kurt-global"
cfg_csv_fp <- "data/kurt/global/llama-mxq-cfgs.csv"
plot_quant_cfg(model_id, budget, attempt, cfg_csv_fp)

attempt <- "kurt-scaled"
cfg_csv_fp <- "data/kurt/scaled/llama-mxq-cfgs.csv"
plot_quant_cfg(model_id, budget, attempt, cfg_csv_fp)
