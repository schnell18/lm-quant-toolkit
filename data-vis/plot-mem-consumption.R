#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(readr)
library(plotly)

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
  "load_mem_allot"
)
all_cols2 <- c(
  "model",
  "algo",
  "config.x",
  "bpp",
  "load_mem_allot"
)
df_all <- read_csv(csv_fp)
df_wo_mxq <- df_all |>
  filter(algo != "mxq") |>
  select(all_of(all_cols1))

df_fp16 <- df_all |>
  filter(algo == "fp16")

df_baseline <- df_all |>
  filter(algo == "hqq") |>
  left_join(df_fp16, by = c("model"), suffix = c(".x", "")) |>
  select(all_of(all_cols2)) |>
  rename(
    config = config.x,
  )

df_hqq <- df_wo_mxq |> filter(algo == "hqq" & !str_detect(config, "^mxq"))
df_mxq <- df_all |> filter(algo == "mxq")
df_mxq1 <- df_hqq |>
  left_join(
    df_mxq,
    suffix = c(".x", ""),
    join_by(model, bpp)
  ) |>
  mutate(
    algo = paste0(algo, "-", attempt)
  ) |>
  select(c("model", "algo", "config.x", "bpp", "load_mem_allot")) |>
  rename(
    config = config.x
  )


df_disp <- bind_rows(df_wo_mxq, df_baseline, df_mxq1) |>
  filter(str_detect(config, "^b4")) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Meta-Llama-3-8B", "Llama-2-13b-hf"),
      labels = c("Llama-2-7B", "Llama-3-8B", "Llama-2-13B")
    ),
    algo = factor(algo, levels = (c("awq", "gptq", "hqq", "mxq-mxq1", "mxq-kurt-global", "mxq-kurt-scaled", "bnb", "fp16"))),
    config = factor(config, levels = (c("b4g32", "b4g64", "b4g128")))
  ) |>
  filter(algo != "bnb")

plt1 <- ggplot(df_disp, aes(x = algo, y = load_mem_allot, fill = algo)) +
  geom_col(aes(x = load_mem_allot, y = algo), show.legend = FALSE) +
  geom_text(
    aes(0, y = algo, label = toupper(algo)),
    hjust = 0,
    nudge_x = 0.3,
    colour = "white",
    size = 3
  ) +
  labs(y = "Algorithm", x = "GPU Memory(GiB)") +
  scale_x_continuous(
    limits = c(0, 22),
    breaks = seq(0, 22, by = 4),
    expand = c(0, 0),
    position = "bottom"
  ) +
  scale_y_discrete(expand = expansion(add = c(0, 0.5))) +
  theme(
    panel.background = element_rect(fill = "white"),
    panel.grid.major.x = element_line(color = "#A8BAC4", size = 0.3),
    axis.ticks.length = unit(0, "mm"),
    axis.title = element_blank(),
    axis.text.y = element_blank()
  ) +
  facet_grid(config ~ model) +
  scale_color_solarized()
plt1
ggsave("pdfs/llama-mem-consumption.pdf", plot = plt1, width = 8, height = 6)

ggplotly(plt1)
