#!/usr/bin/env Rscript

library(dplyr)
library(ggthemes)
library(ggplot2)
library(patchwork)
library(readr)
library(optparse)

load_old <- function(model_id) {
  df_kurt1 <- read_csv(paste0("../src/data/fnorm-", model_id, ".csv")) |>
    group_by(module, layer) |>
    summarise(
      kurtosis = min(kurtosis),
      param_count = min(params)
    )
  df_kurt1$model <- model_id
  df_kurt1$source <- "old"
  return(df_kurt1)
}

load_new <- function(model_id) {
  df_kurt2 <- read_csv(paste0("/tmp/kurtosis-dump/kurtosis-", model_id, ".csv"))
  df_kurt2$model <- model_id
  df_kurt2$source <- "new"
  return(df_kurt2)
}

df_kurt <- bind_rows(
  load_old("Llama-2-7b-hf"),
  load_old("Llama-2-13b-hf"),
  load_new("Llama-2-7b-hf"),
  load_new("Llama-2-13b-hf")
)

line_plot <- ggplot(df_kurt, aes(x = layer, y = kurtosis, color = source, shape = source)) +
  geom_line() +
  theme_gray(base_size = 14) +
  theme_minimal() +
  labs(y = "Kurtosis") +
  theme(
    axis.title.y = element_text(size = 12),
    axis.title.x = element_blank(),
    axis.text.x = element_blank()
  ) +
  facet_grid(model ~ module)
line_plot
