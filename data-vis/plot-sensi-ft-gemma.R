#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggplot2)

df_all <- read_csv("data/qnt-mtd-sensi.csv")

df_layer <- df_all |>
  filter(nbits != 2) |>
  filter(quant_method == "hqq") |>
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
      levels = c("wikitext", "c4", "pileval", "bos"),
      labels = c("WikiText2", "C4", "pileval", "BoS")
    ),
    model = factor(
      model,
      levels = c("gemma-7b", "gemma-7b-it", "codegemma-7b", "codegemma-7b-it"),
      labels = c("gemma-7b", "gemma-7b-it", "codegemma-7b", "codegemma-7b-it")
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
  facet_grid(dataset ~ model, scales = "free") +
  theme(
    strip.background = element_rect(
      color = "darkgray", fill = "white", size = 1.0, linetype = "solid"
    )
  ) +
  scale_color_solarized()
ggsave(
  "pdfs/sensi-ft-llama.pdf",
  plot = plt,
  width = 9,
  height = 6
)
