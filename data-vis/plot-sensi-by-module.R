#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggplot2)

df_all <- read_csv("data/llama-sensitivity.csv")

df_layer <- df_all |>
  # filter(model == "Llama-2-7b-hf") |>
  filter(model == "Meta-Llama-3-8B") |>
  # filter(nbits != 2 & (group_size == 32 | group_size == 128)) |>
  filter((nbits == 4 | nbits == 8)) |>
  mutate(
    cfg = paste0("b", nbits, "g", group_size)
  ) |>
  select(-c("nbits", "group_size")) |>
  mutate(
    dataset = factor(
      dataset,
      levels = c("wikitext", "c4", "pileval", "bos"),
      labels = c("WikiText2", "C4", "pileval", "BoS")
    )
  )

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
  facet_grid(dataset ~ part, scales = "free") +
  theme(
    strip.background = element_rect(
      color = "darkgray", fill = "white", linewidth = 1.0, linetype = "solid"
    )
  ) +
  scale_color_solarized()
ggsave(
  "pdfs/sensi-by-moduel.pdf",
  plot = plt,
  width = 10,
  height = 6
)
