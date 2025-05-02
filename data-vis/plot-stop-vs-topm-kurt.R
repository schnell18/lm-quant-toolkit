#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(tidyverse)
library(readr)
library(openxlsx)
library(optparse)
library(this.path)
library(viridis)

combined_csv_fp <- "endeavors/boost/data/combined.csv"

df_kb <- read_csv(combined_csv_fp) |>
  filter(
    grepl("kurt-boost", attempt)
    # grepl("kurt-boost", attempt) | grepl("sensi-boost", attempt)
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
      labels = c("2 Stops", "3 Stops")
    ),
    topm = factor(
      topm,
      levels = c(1, 2, 3),
      labels = c("Top 1", "Top 2", "Top 3")
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
  filter(topm != 0) |>
  select(c("model", "bpp", "ppl", "dataset", "method", "stop", "topm"))

for (mdl in unique(df_kb$model)) {
  df_plot <- df_kb |> filter(model == mdl)
  plt <- ggplot(df_plot, aes(x = bpp, y = ppl, fill = dataset)) +
    geom_bar(
      aes(fill = dataset),
      stat = "identity", color = "white",
      position = position_dodge(0.9)
    ) +
    geom_text(
      data = subset(df_plot, dataset == "WikiText2"),
      aes(y = ppl, label = formatC(ppl, format = "f", digits = 2)),
      nudge_x = -0.25,
      nudge_y = 0.3,
      size = 2.8
    ) +
    geom_text(
      data = subset(df_plot, dataset == "C4"),
      aes(y = ppl, label = formatC(ppl, format = "f", digits = 2)),
      nudge_x = 0.2,
      nudge_y = 0.3,
      size = 2.8
    ) +
    labs(
      # title = paste0(
      #   "Perplexity of ",
      #   mdl,
      #   " under various boost stops and topm"
      # ),
      x = "Bit Per Parameter",
      y = "Perplexity"
    ) +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(
        color = "darkgray", fill = "white", linewidth = 1.0, linetype = "solid"
      ),
      legend.title = element_blank()
    ) +
    facet_grid(stop ~ topm)
  ggsave(
    create.dir = TRUE,
    plot = plt,
    paste0("pdfs/ppl-stop-vs-topm-", mdl, ".pdf"),
    height = 6,
    width = 10
  )
}
