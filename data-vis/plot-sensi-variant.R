#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  data_dir <- "."
} else {
  data_dir <- args[1]
}

dat_dir <- path.expand(data_dir)
dat_fps <- dir(
  path = dat_dir,
  pattern = ".*\\.csv$",
  recursive = FALSE,
  full.names = TRUE
)

df_all <- plyr::ldply(
  dat_fps,
  read.csv,
  stringsAsFactors = FALSE,
)

write_csv(df_all, "variant-sensi.csv")

df_layer <- df_all |>
  filter(nbits != 2 & (group_size == 32 | group_size == 128)) |>
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
  "pdfs/sensi-variant.pdf",
  plot = plt,
  width = 10,
  height = 6
)
