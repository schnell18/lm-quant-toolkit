library(tidyverse)
library(readr)
library(ggthemes)
library(ggplot2)
library(patchwork)
library(plotly)

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
  df_mod_wdist1 <- df_mod_wdist |> filter(attempt == "PCT5")
  bar_plot1 <- ggplot(
    df_mod_wdist1,
    aes(x = layer, y = abs_val, fill = nth_percentile)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    theme_gray(base_size = 14) +
    labs(
      x = df_mod_wdist1$mod_disp[1],
      y = "Absolute Value",
      fill = "nth percentile"
    ) +
    geom_text(
      data = subset(df_mod_wdist1, nth_percentile == 100),
      aes(x = layer, label = quant_cfg),
      angle = 90,
      vjust = 0.20,
      position = position_stack(vjust = 0.5),
      colour = "white",
      size = 2
    ) +
    theme(legend.position = "none") +
    scale_color_solarized()

  df_mod_wdist2 <- df_mod_wdist |> filter(attempt == "PCT6")
  bar_plot2 <- ggplot(
    df_mod_wdist2,
    aes(x = layer, y = abs_val, fill = nth_percentile)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    theme_gray(base_size = 14) +
    labs(
      x = df_mod_wdist2$mod_disp[1],
      y = "Absolute Value",
      fill = "nth percentile"
    ) +
    geom_text(
      data = subset(df_mod_wdist2, nth_percentile == 100),
      aes(x = layer, label = quant_cfg),
      angle = 90,
      vjust = 0.20,
      position = position_stack(vjust = 0.5),
      colour = "white",
      size = 2
    ) +
    theme(legend.position = "none") +
    scale_color_solarized()

  df_mod_wdist3 <- df_mod_wdist |> filter(attempt == "kurt-scaled")
  bar_plot3 <- ggplot(
    df_mod_wdist3,
    aes(x = layer, y = abs_val, fill = nth_percentile)
  ) +
    geom_bar(stat = "identity", color = "gray50") +
    theme_gray(base_size = 14) +
    labs(
      x = df_mod_wdist3$mod_disp[1],
      y = "Absolute Value",
      fill = "nth percentile"
    ) +
    geom_text(
      data = subset(df_mod_wdist3, nth_percentile == 100),
      aes(x = layer, label = quant_cfg),
      angle = 90,
      vjust = 0.20,
      position = position_stack(vjust = 0.5),
      colour = "white",
      size = 2
    )
  if (show_legend) {
    bar_plot3 <- bar_plot3 +
      theme(
        legend.position = "bottom",
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16)
      ) +
      # coord_flip() +
      scale_color_solarized()
  } else {
    bar_plot3 <- bar_plot3 +
      theme(legend.position = "none") +
      scale_color_solarized()
  }

  # Combine the line and bar plot vertically
  combined_plot <- line_plot / bar_plot1 / bar_plot2 / bar_plot3 + plot_layout(heights = c(1, 3, 3, 3))
  return(combined_plot)
}

weight_grid_only <- function(
    df_wdist, df_kurtosis, mod, show_legend = FALSE) {
  return(weight_grid(df_wdist, df_kurtosis, mod, show_legend, show_cfg = FALSE))
}

budget <- 4.51
model_id <- "Llama-2-13b-hf"
df_cfg1 <- read_csv("data/mxq-quant-cfgs-mxq1-5pct-tol.csv")
df_cfg2 <- read_csv("data/mxq-quant-cfgs-kurt-scaled-6pct-tol.csv")
df_cfg3 <- read_csv("data/kurt/scaled/llama-mxq-cfgs.csv")
df_cfg1$attempt <- "PCT5"
df_cfg2$attempt <- "PCT6"
df_cfg3$attempt <- "kurt-scaled"
df_cfg <- bind_rows(df_cfg1, df_cfg2, df_cfg3)

df_cfg_1 <- df_cfg |>
  filter(bit_budget == budget & model == model_id) |>
  mutate(
    quant_cfg = paste0("b", b1, "g", g1)
  ) |>
  select(-c("b1", "g1", "b2", "g2", "bit_budget"))

df_all <- read_csv(paste0("data/wdist/wdist-", model_id, ".csv"))
percentiles <- c("0", "99", "99.9", "99.99", "100")
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

df_cfg_1 <- df_cfg_1 |>
  left_join(df_module_param_count, by = c("module")) |>
  mutate(
    mod_disp = paste0(attempt, " ", mod_disp)
  )

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
  filter(!grepl("_layernorm", module)) |>
  left_join(df_cfg_1, by = c("module", "layer"), relationship = "many-to-many")


k_cols <- c("module", "layer", "kurtosis")
df_kurtosis <- df_all |>
  select(all_of(k_cols))

p1 <- weight_grid(df_wdist, df_kurtosis, "mlp.down_proj")
p2 <- weight_grid(df_wdist, df_kurtosis, "mlp.gate_proj")
p3 <- weight_grid(df_wdist, df_kurtosis, "mlp.up_proj")
p4 <- weight_grid(df_wdist, df_kurtosis, "self_attn.k_proj")
p5 <- weight_grid(df_wdist, df_kurtosis, "self_attn.o_proj")
p6 <- weight_grid(df_wdist, df_kurtosis, "self_attn.q_proj", TRUE)
p7 <- weight_grid(df_wdist, df_kurtosis, "self_attn.v_proj")

final_plot1 <- (p1 | p2)
final_plot1
ggsave(
  paste0("pdfs/", model_id, "-mxq-cfgs-from-model1.pdf"),
  plot = final_plot1, width = 16, height = 9
)

final_plot2 <- (p3 | p3)
final_plot2
ggsave(
  paste0("pdfs/", model_id, "-mxq-cfgs-from-model2.pdf"),
  plot = final_plot2, width = 11, height = 6
)

final_plot3 <- (p4 | p5)
final_plot3
ggsave(
  paste0("pdfs/", model_id, "-mxq-cfgs-from-model3.pdf"),
  plot = final_plot3, width = 11, height = 6
)

final_plot4 <- (p6 | p7)
final_plot4
ggsave(
  paste0("pdfs/", model_id, "-mxq-cfgs-from-model4.pdf"),
  plot = final_plot4, width = 11, height = 6
)
