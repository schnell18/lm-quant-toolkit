library(plyr)
library(dplyr)
library(tidyverse)
library(ggplot2)

calc_bpp <- function(config) {
  if (config == "base") {
    return(16.0)
  } else if (startsWith(config, "b")) {
    b1 <- strtoi(substr(config, 2, 2))
    g1 <- strtoi(substr(config, 4, nchar(config)))
    b2 <- 8
    g2 <- 128
    return(round(b1 + 2 * b2 / g1 + 32 / g1 / g2, digits = 2))
  } else {
    return(round(as.numeric(sub("_", ".", config)), digits = 2))
  }
}

kurt_dir <- path.expand("data/kurt/global")
kurt_fps <- dir(
  path = kurt_dir,
  pattern = "result-eval_ppl-kurt-.*\\.csv$",
  full.names = TRUE
)
df_kurt <- ldply(kurt_fps, read.csv, stringsAsFactors = FALSE)
df_kurt$attempt <- "kurt-global"

kurt_scaled_dir <- path.expand("data/kurt/scaled/")
kurt_scaled_fps <- dir(
  path = kurt_scaled_dir,
  pattern = "result-eval_ppl-kurt-scaled-.*\\.csv$",
  full.names = TRUE
)
df_kurt_scaled <- ldply(kurt_scaled_fps, read.csv, stringsAsFactors = FALSE)
df_kurt_scaled$attempt <- "kurt-scaled"

base_dir <- "data/"
base_fps <- dir(
  path = base_dir,
  pattern = "result-eval_ppl.*mxq.*\\.csv$",
  full.names = TRUE
)
df_base <- ldply(base_fps, read.csv, stringsAsFactors = FALSE) |>
  filter(
    config == "4_51" |
      config == "4_25" |
      config == "4_13" |
      config == "3_51" |
      config == "3_25" |
      config == "3_13"
  )
df_base$attempt <- "MXQ1"

hqq_dir <- "data/"
hqq_fps <- dir(
  path = hqq_dir,
  pattern = "result-eval_ppl_hqq.*\\.csv$",
  full.names = TRUE
)
df_hqq <- ldply(hqq_fps, read.csv, stringsAsFactors = FALSE) |>
  filter(
    config == "b3g32" |
      config == "b3g64" |
      config == "b3g128" |
      config == "b4g32" |
      config == "b4g64" |
      config == "b4g128"
  )
df_hqq$attempt <- "HQQ"

df_all <- bind_rows(df_base, df_kurt, df_kurt_scaled, df_hqq) |>
  select(
    c(
      "model",
      "algo",
      "attempt",
      "config",
      "ppl_wikitext",
      "ppl_c4",
      "ppl_mem_allot"
    )
  ) |>
  mutate(
    bpp = sapply(config, calc_bpp),
    ppl_mem_allot = round(ppl_mem_allot / 1024**3, digits = 2)
  ) |>
  pivot_longer(
    cols = c("ppl_wikitext", "ppl_c4"),
    names_to = c(".value", "dataset"),
    names_sep = "_"
  )


ggplot(
  data = subset(df_all, model != "Meta-Llama-3-8B"),
  aes(x = bpp, y = ppl)
) +
  geom_line(aes(color = attempt, y = ppl)) +
  geom_point(aes(shape = attempt, color = attempt, y = ppl)) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 16) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16)
  ) +
  facet_grid(dataset ~ model, scales = "free")

ggplot(
  data = subset(df_all, model == "Meta-Llama-3-8B"),
  aes(x = bpp, y = ppl)
) +
  geom_line(aes(color = attempt, y = ppl)) +
  geom_point(aes(shape = attempt, color = attempt, y = ppl)) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 16) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16)
  ) +
  facet_grid(dataset ~ model, scales = "free")

