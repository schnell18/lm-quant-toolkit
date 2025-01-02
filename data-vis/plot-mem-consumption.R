#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(readr)
library(plotly)
library(optparse)

second_largest <- function(x) {
  sort(unique(x), decreasing = TRUE)[2L]
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--csv_file"),
  type = "character",
  help = "Combined PPL result CSV file",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt"),
  type = "character",
  help = "The attempt to plot",
  metavar = "character"
)

args <- parse_args(parser)

if (is.null(args$csv_file)) {
  csv_fp <- "data/combined.csv"
} else {
  csv_fp <- args$csv_file
}
if (is.null(args$attempt)) {
  the_attempt <- "mxq1"
} else {
  the_attempt <- args$attempt
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
df_mxq <- df_all |>
  filter(
    algo == "mxq" & attempt != "mxq-kurt-global" & attempt != "mxq-kurt-scaled"
  )
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
    algo = factor(
      algo,
      levels = (
        c(
          the_attempt,
          "awq",
          "gptq",
          "hqq",
          "mxq-mxq1",
          "mxq-mxq2",
          "bnb",
          "fp16"
        )
      )
    ),
    config = factor(config, levels = (c("b4g32", "b4g64", "b4g128")))
  ) |>
  filter(algo != "bnb")

df_2nd_largest <- df_disp |>
  group_by(model) |>
  summarise(
    second_max_mem = second_largest(load_mem_allot)
  ) |>
  ungroup()
df_disp <- df_disp |>
  left_join(
    df_2nd_largest,
    join_by(model)
  )

plt1 <- ggplot(df_disp, aes(x = algo, y = load_mem_allot, fill = algo)) +
  geom_col(aes(x = load_mem_allot, y = algo), show.legend = FALSE) +
  geom_text(
    data = subset(df_disp, algo == "fp16"),
    aes(second_max_mem, y = algo, label = toupper(algo)),
    hjust = 0,
    nudge_x = -0.5,
    colour = "white",
    size = 3
  ) +
  geom_text(
    data = subset(df_disp, algo != "fp16"),
    aes(x = load_mem_allot, y = algo, label = toupper(algo)),
    hjust = 0,
    nudge_x = 0.3,
    colour = "black",
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
