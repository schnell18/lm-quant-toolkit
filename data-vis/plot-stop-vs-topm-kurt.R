#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(tidyverse)
library(readr)

combined_csv_fp <- "endeavors/boost/data/combined.csv"

df_kb <- read_csv(combined_csv_fp) |>
  filter(
    grepl("kurt-boost", attempt)
  ) |>
  separate_wider_regex(
    attempt,
    c(method = "\\w+-\\w+", "-", stop = "\\d", "-", topm = "\\d")
  ) |>
  pivot_longer(
    cols = c("ppl_wikitext", "ppl_c4"),
    names_to = c(".value", "dataset"),
    names_sep = "_"
  ) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
      labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
    ),
    stop = factor(
      stop,
      levels = c(2, 3),
      labels = c("2 Boost Levels", "3 Boost Levels")
    ),
    topm = factor(
      topm,
      levels = c(1, 2, 3, 0),
      labels = c("1", "2", "3", "Max")
    ),
    bpp = factor(
      bpp,
      levels = c(3.13, 3.25, 3.51, 4.13, 4.25, 4.51),
      labels = c("3.13", "3.25", "3.51", "4.13", "4.25", "4.51")
    ),
    dataset = factor(
      dataset,
      levels = c("wikitext", "c4"),
      labels = c("WikiText2", "C4")
    ),
  ) |>
  select(c("model", "bpp", "ppl", "dataset", "method", "stop", "topm"))

df_plot <- df_kb |>
  filter(bpp == 4.13 | bpp == 4.25 | bpp == 4.51) |>
  filter(dataset == "WikiText2")
# filter(model == "Llama-2-13B")

plt <- ggplot(df_plot, aes(x = topm, y = ppl)) +
  geom_point(aes(shape = stop, color = stop)) +
  geom_line(aes(group = stop, color = stop)) +
  scale_color_manual(
    values = c(
      "2 Boost Levels" = "#66c2a5",
      "3 Boost Levels" = "#fc8d62"
    )
  ) +
  labs(
    x = "Topm",
    y = "Perplexity"
  ) +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(
      color = "darkgray", fill = "white", linewidth = 1.0, linetype = "solid"
    ),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    legend.title = element_blank()
  ) +
  scale_y_continuous(
    labels = function(y) format(y, nsmall = 3, scientific = FALSE)
  ) +
  facet_grid(model ~ bpp, scales = "free")
# facet_grid(dataset ~ bpp, scales = "free")

ggsave(
  create.dir = TRUE,
  plot = plt,
  paste0("pdfs/ppl-stop-vs-topm.pdf"),
  height = 6,
  width = 7
)
