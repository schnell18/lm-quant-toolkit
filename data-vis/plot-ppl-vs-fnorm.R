#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(readr)

all_cols <- c(
  "model", "algo", "config",
  "bpp", "ppl_wikitext", "ppl_c4", "fnorm"
)

models <- c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B")
model_labels <- c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
df_obj <- read_csv("data/mxq-objectives.csv")


fnorms <- list()
for (model in models) {
  base_dir <- "data/fnorm"
  csv_fp <- file.path(base_dir, paste0("fnorm-", model, ".csv"))
  df <- read.csv(csv_fp) |>
    filter(
      (nbit1 == 8 & gsize1 == 32) |
        (nbit1 == 8 & gsize1 == 64) |
        (nbit1 == 8 & gsize1 == 128) |
        (nbit1 == 4 & gsize1 == 32) |
        (nbit1 == 4 & gsize1 == 64) |
        (nbit1 == 4 & gsize1 == 128) |
        (nbit1 == 3 & gsize1 == 32) |
        (nbit1 == 3 & gsize1 == 64) |
        (nbit1 == 3 & gsize1 == 128) |
        (nbit1 == 2 & gsize1 == 16) |
        (nbit1 == 2 & gsize1 == 32) |
        (nbit1 == 2 & gsize1 == 64) |
        (nbit1 == 2 & gsize1 == 128)
    ) |>
    summarise(
      .by = c("nbit1", "gsize1", "nbit2", "gsize2"),
      fnorm = sum(fnorm)
    ) |>
    mutate(
      bpp = round(nbit1 + 2 * nbit2 / gsize1 + 32 / gsize1 / gsize2, digits = 2),
      .before = 0
    ) |>
    select(!c("nbit1", "gsize1", "nbit2", "gsize2")) |>
    add_column(model = model, algo = "hqq", .before = 0)

  fnorms <- append(fnorms, list(df))
}
df_fnorm <- bind_rows(fnorms)

df_all <- read_csv("data/combined.csv")
df_mxq <- df_all |>
  filter(algo == "mxq") |>
  left_join(df_obj, by = c("model", "bpp"))

df_hqq <- df_all |>
  filter(algo == "hqq") |>
  left_join(df_fnorm, by = c("model", "algo", "bpp"))

dfs <- list()
dfs <- append(dfs, list(df_hqq))
dfs <- append(dfs, list(df_mxq))
df_disp <- bind_rows(dfs) |>
  filter(bpp >= 3.00) |>
  select(all_of(all_cols)) |>
  mutate(
    model = factor(
      model,
      levels = models,
      labels = model_labels
    ),
  ) |>
  pivot_longer(
    cols = c("ppl_wikitext", "ppl_c4"),
    names_to = c(".value", "dataset"),
    names_sep = "_"
  ) |>
  mutate(
    dataset = factor(
      dataset,
      levels = c("wikitext", "c4"),
      labels = c("WikiText2", "C4")
    ),
    algo = factor(
      algo,
      levels = c("mxq", "hqq"),
      labels = c("MXQ", "HQQ")
    )
  ) |>
  filter(model == "Llama-2-13B")

# Plot FNorm vs Perplexity ---------------------------------
plt1 <- ggplot(subset(df_disp, algo == "MXQ"), aes(x = fnorm, y = ppl)) +
  geom_point(size = 0.2, aes(color = dataset, shape = algo)) +
  geom_point(
    data = subset(df_disp, algo == "HQQ"),
    size = 1.2, aes(shape = algo, color = dataset, x = fnorm, y = ppl)
  ) +
  geom_text(
    data = subset(df_disp, algo == "HQQ" & bpp < 4.13),
    aes(x = fnorm, label = config),
    hjust = 1.2,
    vjust = 0.7,
    size = 2
  ) +
  geom_text(
    data = subset(df_disp, algo == "HQQ" & bpp >= 4.13 & bpp <= 8.13),
    aes(x = fnorm, label = config),
    hjust = -0.2,
    vjust = -1.0,
    size = 2
  ) +
  labs(x = "FNorm", y = "Perplexity") +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  guides(shape = FALSE) +
  facet_wrap(~model, ncol = 1, scales = "free") +
  scale_color_solarized()
plt1
ggsave("pdfs/mxq-ppl-vs-fnorm.pdf", plot = plt1, width = 8, height = 5)

# Plot BitBudget vs FNorm ---------------------------------
plt2 <- ggplot(subset(df_disp, algo == "MXQ"), aes(x = bpp, y = fnorm)) +
  geom_point(size = 0.4, aes(color = dataset, shape = algo)) +
  geom_point(
    data = subset(df_disp, algo == "HQQ"),
    size = 0.4, aes(shape = algo, color = dataset)
  ) +
  geom_text(
    data = subset(df_disp, algo == "HQQ"),
    aes(x = fnorm, label = config),
    vjust = 1,
    size = 2
  ) +
  labs(x = "Bit Budget", y = "FNorm") +
  theme_gray(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, ncol = 1, scales = "free") +
  scale_color_solarized()
plt2

# Plot BitBudget vs Perplexiy ---------------------------------
plt3 <- ggplot(subset(df_disp, algo == "MXQ"), aes(x = bpp, y = ppl)) +
  geom_point(size = 0.2, aes(color = dataset, shape = algo, x = bpp, y = ppl)) +
  geom_point(
    data = subset(df_disp, algo == "HQQ"),
    size = 1.2, aes(shape = algo, color = dataset, x = bpp, y = ppl)
  ) +
  geom_text(
    data = subset(df_disp, algo == "HQQ" & bpp < 4.13),
    aes(x = bpp, label = config),
    hjust = 1.2,
    vjust = 0.7,
    size = 2.5
  ) +
  geom_text(
    data = subset(df_disp, algo == "HQQ" & bpp == 4.13),
    aes(x = bpp, label = config),
    hjust = 1.2,
    vjust = -1.0,
    size = 2.5
  ) +
  geom_text(
    data = subset(df_disp, algo == "HQQ" & bpp > 4.13 & bpp != 8.13),
    aes(x = bpp, label = config),
    hjust = -0.2,
    vjust = -1.0,
    size = 2.5
  ) +
  geom_text(
    data = subset(df_disp, algo == "HQQ" & bpp == 8.13),
    aes(x = bpp, label = config),
    hjust = 1.0,
    vjust = -1.0,
    size = 2.5
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  guides(shape = FALSE) +
  facet_wrap(~model, ncol = 1, scales = "free") +
  scale_color_solarized()
plt3
ggsave("pdfs/mxq-bpp-vs-ppl.pdf", plot = plt3, width = 8, height = 8)
