#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  csv_fp <- "data/combined.csv"
} else {
  csv_fp <- args[1]
}

all_cols1 <- c(
  "model",
  "algo",
  "config",
  "bpp",
  "quant_duration"
)
df_all <- read_csv(csv_fp)
df_wo_mxq <- df_all |>
  filter(algo != "mxq" & algo != "fp16" & algo != "bnb") |>
  select(all_of(all_cols1))

df_hqq <- df_all |> filter(algo == "hqq" & !str_detect(config, "^mxq"))
df_mxq <- df_all |>
  filter(algo == "mxq" & attempt == "mxq1") |>
  select(all_of(all_cols1))

df_mxq1 <- df_hqq |>
  left_join(
    df_mxq,
    suffix = c(".x", ""),
    join_by(model, bpp)
  ) |>
  select(c("model", "algo", "config.x", "bpp", "quant_duration")) |>
  rename(
    config = config.x
  )

df_all <- bind_rows(df_wo_mxq, df_mxq1)

duration_cutoff <- 10
disp <- df_all |>
  filter(str_detect(config, "^b4")) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Meta-Llama-3-8B", "Llama-2-13b-hf"),
      labels = c("Llama-2-7B", "Llama-3-8B", "Llama-2-13B")
    ),
    config = factor(config, levels = (c("b4g32", "b4g64", "b4g128")))
  )

ggplot(disp, aes(x = algo, y = quant_duration, fill = algo)) +
  geom_col(aes(x = quant_duration, y = algo), show.legend = FALSE) +
  geom_text(
    data = subset(disp, quant_duration <= duration_cutoff),
    aes(quant_duration * 1.1, y = algo, label = toupper(algo)),
    hjust = 0,
    nudge_x = 0.3,
    size = 3
  ) +
  geom_text(
    data = subset(disp, quant_duration > duration_cutoff),
    aes(0, y = algo, label = toupper(algo)),
    hjust = -0.2,
    nudge_x = 0.3,
    colour = "white",
    size = 3
  ) +
  labs(y = "Algorithm", x = "Quantation Time(Seconds)") +
  scale_x_continuous(
    limits = c(1, 1600),
    expand = c(0, 0),
    trans = "log10",
    position = "bottom"
  ) +
  scale_y_discrete(expand = expansion(add = c(0, 0.5))) +
  theme(
    panel.background = element_rect(fill = "white"),
    panel.grid.major.x = element_line(color = "#A8BAC4", size = 0.2),
    axis.ticks.length = unit(0, "mm"),
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 30, vjust = 0.9, hjust = 1),
    axis.text.y = element_blank()
  ) +
  # facet_grid(config ~ model, scales = "free" ) +
  facet_grid(config ~ model) +
  scale_color_tableau()
ggsave("pdfs/llama-quant-speed.pdf", width = 10, height = 4)
