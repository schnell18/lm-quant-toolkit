#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  csv_fp <- "data/variant-sensi.csv"
} else {
  csv_fp <- args[1]
}

df_all <- read_csv(csv_fp)

df_layer <- df_all |>
  group_by(dataset, nbits, group_size, model, layer) |>
  summarise(
    sensitivity = sum(sensitivity)
  ) |>
  ungroup() |>
  mutate(
    cfg = paste0("b", nbits, "g", group_size)
  ) |>
  select(-c("nbits", "group_size")) |>
  mutate(
    dataset = factor(
      dataset,
      levels = c("wikitext", "c4", "pileval"),
      labels = c("WikiText2", "C4", "pileval")
    )
  )


# Llama-2-7b-chat-hf
# Llama-2-7b-hf
# meditron-7b

plt <- ggplot(df_layer, aes(x = layer, y = sensitivity)) +
  geom_point(
    size = 1.5,
    aes(shape = cfg, color = cfg)
  ) +
  geom_line(
    linewidth = 0.5,
    aes(color = cfg)
  ) +
  labs(x = "Layer", y = "Sensitivity") +
  scale_x_continuous(
    breaks = seq(0, 40, 5)
  ) +
  scale_y_continuous(trans = "log10") +
  theme_gray(base_size = 14) +
  theme(
    legend.position = "bottom"
  ) +
  guides(color = guide_legend(nrow = 1)) +
  facet_grid(dataset ~ model, scales = "free") +
  scale_color_solarized()
ggsave(
  "pdfs/variant-sensi.pdf",
  plot = plt,
  width = 10,
  height = 6
)

