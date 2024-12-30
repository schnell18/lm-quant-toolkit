#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(tidyverse)
library(readr)
library(openxlsx)
library(optparse)
library(this.path)

# make reference to library function portable
source(file.path(here("functions"), "utils.R"))

abbr_method <- function(method) {
  if (method == "SensiBoost") {
    return("SB")
  } else if (method == "KurtBoost") {
    return("KB")
  } else if (method == "Ablation") {
    return("ABL")
  }
  return(method)
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-f", "--ppl_csv_file"),
  type = "character",
  help = "Combined PPL metrics CSV file",
  metavar = "character"
)

args <- parse_args(parser)

if (is.null(args$ppl_csv_file)) {
  csv_file <- "data/combined.csv"
} else {
  csv_file <- args$ppl_csv_file
}

df_all <- read_csv(csv_file)

df_hqq <- df_all |>
  filter(
    algo == "hqq"
  )

df_mxq <- df_all |>
  filter(
    algo == "mxq" & attempt == "mxq1"
  ) |>
  filter(
    bpp == 3.13 |
      bpp == 3.25 |
      bpp == 3.51 |
      bpp == 4.13 |
      bpp == 4.25 |
      bpp == 4.51
  )

# sensi-boost vs HQQ ------------------------------------------------------
df_sb <- df_all |>
  filter(
    grepl("sensi-boost", attempt)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4"
    )
  ) |>
  mutate(
    attempt = sapply(attempt, abbrev_sensi_kurt)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = c(ppl_wikitext, ppl_c4),
    names_vary = "slowest"
  ) |>
  pivot_longer(
    cols = !c("model", "algo", "config", "bpp"),
    names_to = c(".value", "setting"),
    names_pattern = "^(.*)(\\d\\d$)",
  ) |>
  left_join(
    df_hqq,
    suffix = c("", "_hqq"),
    by = c("model", "bpp")
  ) |>
  mutate(
    ppl_wikitext_SB = round(ppl_wikitext_SB, digits = 2),
    ppl_c4_SB = round(ppl_c4_SB, digits = 2),
    ppl_wikitext = round(ppl_wikitext, digits = 2),
    ppl_c4 = round(ppl_c4, digits = 2)
  ) |>
  mutate(
    score_wk = round(ppl_wikitext_SB - ppl_wikitext, digits = 4),
    score_c4 = round(ppl_c4_SB - ppl_c4, digits = 4)
  ) |>
  mutate(
    wins = ifelse(score_wk < 0, 1, 0) + ifelse(score_c4 < 0, 1, 0),
    ties = ifelse(score_wk == 0, 1, 0) + ifelse(score_c4 == 0, 1, 0),
    losses = ifelse(score_wk > 0, 1, 0) + ifelse(score_c4 > 0, 1, 0)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "setting",
      "wins",
      "losses",
      "ties",
      "ppl_wikitext_SB",
      "ppl_c4_SB",
      "ppl_wikitext",
      "ppl_c4"
    )
  )

df_sb_sum <- df_sb |>
  group_by(model) |>
  summarise(
    wins = sum(wins),
    ties = sum(ties),
    losses = sum(losses)
  ) |>
  ungroup()
df_sb_sum$Method1 <- "sensi-boost"
df_sb_sum$Method2 <- "HQQ"

# sensi-boost vs MXQ ------------------------------------------------------
df_sb_mxq <- df_all |>
  filter(
    grepl("sensi-boost", attempt)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4"
    )
  ) |>
  mutate(
    attempt = sapply(attempt, abbrev_sensi_kurt)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = c(ppl_wikitext, ppl_c4),
    names_vary = "slowest"
  ) |>
  pivot_longer(
    cols = !c("model", "algo", "config", "bpp"),
    names_to = c(".value", "setting"),
    names_pattern = "^(.*)(\\d\\d$)",
  ) |>
  left_join(
    df_mxq,
    suffix = c("", "_mxq"),
    by = c("model", "bpp")
  ) |>
  mutate(
    ppl_wikitext_SB = round(ppl_wikitext_SB, digits = 2),
    ppl_c4_SB = round(ppl_c4_SB, digits = 2),
    ppl_wikitext = round(ppl_wikitext, digits = 2),
    ppl_c4 = round(ppl_c4, digits = 2)
  ) |>
  mutate(
    score_wk = round(ppl_wikitext_SB - ppl_wikitext, digits = 4),
    score_c4 = round(ppl_c4_SB - ppl_c4, digits = 4)
  ) |>
  mutate(
    wins = ifelse(score_wk < 0, 1, 0) + ifelse(score_c4 < 0, 1, 0),
    ties = ifelse(score_wk == 0, 1, 0) + ifelse(score_c4 == 0, 1, 0),
    losses = ifelse(score_wk > 0, 1, 0) + ifelse(score_c4 > 0, 1, 0)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "setting",
      "wins",
      "losses",
      "ties",
      "ppl_wikitext_SB",
      "ppl_c4_SB",
      "ppl_wikitext",
      "ppl_c4"
    )
  )

df_sb_mxq_sum <- df_sb_mxq |>
  group_by(model) |>
  summarise(
    wins = sum(wins),
    ties = sum(ties),
    losses = sum(losses)
  ) |>
  ungroup()
df_sb_mxq_sum$Method1 <- "sensi-boost"
df_sb_mxq_sum$Method2 <- "MXQ"

# kurt-boost vs MXQ ------------------------------------------------------
df_kb_mxq <- df_all |>
  filter(
    grepl("kurt-boost", attempt)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4"
    )
  ) |>
  mutate(
    attempt = sapply(attempt, abbrev_sensi_kurt)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = c(ppl_wikitext, ppl_c4),
    names_vary = "slowest"
  ) |>
  pivot_longer(
    cols = !c("model", "algo", "config", "bpp"),
    names_to = c(".value", "setting"),
    names_pattern = "^(.*)(\\d\\d$)",
  ) |>
  left_join(
    df_mxq,
    suffix = c("", "_mxq"),
    by = c("model", "bpp")
  ) |>
  mutate(
    ppl_wikitext_KB = round(ppl_wikitext_KB, digits = 2),
    ppl_c4_KB = round(ppl_c4_KB, digits = 2),
    ppl_wikitext = round(ppl_wikitext, digits = 2),
    ppl_c4 = round(ppl_c4, digits = 2)
  ) |>
  mutate(
    score_wk = round(ppl_wikitext_KB - ppl_wikitext, digits = 4),
    score_c4 = round(ppl_c4_KB - ppl_c4, digits = 4)
  ) |>
  mutate(
    wins = ifelse(score_wk < 0, 1, 0) + ifelse(score_c4 < 0, 1, 0),
    ties = ifelse(score_wk == 0, 1, 0) + ifelse(score_c4 == 0, 1, 0),
    losses = ifelse(score_wk > 0, 1, 0) + ifelse(score_c4 > 0, 1, 0)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "setting",
      "wins",
      "losses",
      "ties",
      "ppl_wikitext_KB",
      "ppl_c4_KB",
      "ppl_wikitext",
      "ppl_c4"
    )
  )

df_kb_mxq_sum <- df_kb_mxq |>
  group_by(model) |>
  summarise(
    wins = sum(wins),
    ties = sum(ties),
    losses = sum(losses)
  ) |>
  ungroup()
df_kb_mxq_sum$Method1 <- "kurt-boost"
df_kb_mxq_sum$Method2 <- "MXQ"

# kurt-boost vs HQQ ------------------------------------------------------

df_kb <- df_all |>
  filter(
    grepl("kurt-boost", attempt)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4"
    )
  ) |>
  mutate(
    attempt = sapply(attempt, abbrev_sensi_kurt)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = c(ppl_wikitext, ppl_c4),
    names_vary = "slowest"
  ) |>
  pivot_longer(
    cols = !c("model", "algo", "config", "bpp"),
    names_to = c(".value", "setting"),
    names_pattern = "^(.*)(\\d\\d$)",
  ) |>
  left_join(
    df_hqq,
    suffix = c("", "_hqq"),
    by = c("model", "bpp")
  ) |>
  mutate(
    ppl_wikitext_KB = round(ppl_wikitext_KB, digits = 2),
    ppl_c4_KB = round(ppl_c4_KB, digits = 2),
    ppl_wikitext = round(ppl_wikitext, digits = 2),
    ppl_c4 = round(ppl_c4, digits = 2)
  ) |>
  mutate(
    score_wk = round(ppl_wikitext_KB - ppl_wikitext, digits = 4),
    score_c4 = round(ppl_c4_KB - ppl_c4, digits = 4)
  ) |>
  mutate(
    wins = ifelse(score_wk < 0, 1, 0) + ifelse(score_c4 < 0, 1, 0),
    ties = ifelse(score_wk == 0, 1, 0) + ifelse(score_c4 == 0, 1, 0),
    losses = ifelse(score_wk > 0, 1, 0) + ifelse(score_c4 > 0, 1, 0)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "setting",
      "wins",
      "losses",
      "ties",
      "ppl_wikitext_KB",
      "ppl_c4_KB",
      "ppl_wikitext",
      "ppl_c4"
    )
  )

df_kb_sum <- df_kb |>
  group_by(model) |>
  summarise(
    wins = sum(wins),
    ties = sum(ties),
    losses = sum(losses)
  ) |>
  ungroup()
df_kb_sum$Method1 <- "kurt-boost"
df_kb_sum$Method2 <- "HQQ"

# kurt-boost vs kurt-boost ablation ----------------------------------------
df_kbab <- df_all |>
  filter(
    grepl("kurt-boost", attempt) | grepl("kurt-abl", attempt)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4"
    )
  ) |>
  mutate(
    attempt = sapply(attempt, abbrev_sensi_kurt)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = c(ppl_wikitext, ppl_c4),
    names_vary = "slowest"
  ) |>
  pivot_longer(
    cols = !c("model", "algo", "config", "bpp"),
    names_to = c(".value", "setting"),
    names_pattern = "^(.*)(\\d\\d$)",
  ) |>
  mutate(
    ppl_wikitext_KB = round(ppl_wikitext_KB, digits = 2),
    ppl_c4_KB = round(ppl_c4_KB, digits = 2),
    ppl_wikitext_KBAB = round(ppl_wikitext_KBAB, digits = 2),
    ppl_c4_KBAB = round(ppl_c4_KBAB, digits = 2)
  ) |>
  mutate(
    score_wk = round(ppl_wikitext_KB - ppl_wikitext_KBAB, digits = 4),
    score_c4 = round(ppl_c4_KB - ppl_c4_KBAB, digits = 4)
  ) |>
  mutate(
    wins = ifelse(score_wk < 0, 1, 0) + ifelse(score_c4 < 0, 1, 0),
    ties = ifelse(score_wk == 0, 1, 0) + ifelse(score_c4 == 0, 1, 0),
    losses = ifelse(score_wk > 0, 1, 0) + ifelse(score_c4 > 0, 1, 0)
  ) |>
  filter(
    !is.na(wins)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "setting",
      "wins",
      "losses",
      "ties",
      "ppl_wikitext_KB",
      "ppl_c4_KB",
      "ppl_wikitext_KBAB",
      "ppl_c4_KBAB"
    )
  )

write.xlsx(df_kbab, "df_kbab.xlsx", asTable = TRUE, overwrite = TRUE)

df_kbab_sum <- df_kbab |>
  group_by(model) |>
  summarise(
    wins = sum(wins),
    ties = sum(ties),
    losses = sum(losses)
  ) |>
  ungroup()
df_kbab_sum$Method1 <- "kurt-boost"
df_kbab_sum$Method2 <- "ablation"

# sensi-boost vs sensi-boost ablation ----------------------------------------
df_sbab <- df_all |>
  filter(
    grepl("sensi-boost", attempt) | grepl("sensi-abl", attempt)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4"
    )
  ) |>
  mutate(
    attempt = sapply(attempt, abbrev_sensi_kurt)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = c(ppl_wikitext, ppl_c4),
    names_vary = "slowest"
  ) |>
  pivot_longer(
    cols = !c("model", "algo", "config", "bpp"),
    names_to = c(".value", "setting"),
    names_pattern = "^(.*)(\\d\\d$)",
  ) |>
  mutate(
    ppl_wikitext_SB = round(ppl_wikitext_SB, digits = 2),
    ppl_c4_SB = round(ppl_c4_SB, digits = 2),
    ppl_wikitext_SBAB = round(ppl_wikitext_SBAB, digits = 2),
    ppl_c4_SBAB = round(ppl_c4_SBAB, digits = 2)
  ) |>
  mutate(
    score_wk = round(ppl_wikitext_SB - ppl_wikitext_SBAB, digits = 4),
    score_c4 = round(ppl_c4_SB - ppl_c4_SBAB, digits = 4)
  ) |>
  mutate(
    wins = ifelse(score_wk < 0, 1, 0) + ifelse(score_c4 < 0, 1, 0),
    ties = ifelse(score_wk == 0, 1, 0) + ifelse(score_c4 == 0, 1, 0),
    losses = ifelse(score_wk > 0, 1, 0) + ifelse(score_c4 > 0, 1, 0)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "setting",
      "wins",
      "losses",
      "ties",
      "ppl_wikitext_SB",
      "ppl_c4_SB",
      "ppl_wikitext_SBAB",
      "ppl_c4_SBAB"
    )
  )

write.xlsx(df_sbab, "df_sbab.xlsx", asTable = TRUE, overwrite = TRUE)

df_sbab_sum <- df_sbab |>
  group_by(model) |>
  summarise(
    wins = sum(wins),
    ties = sum(ties),
    losses = sum(losses)
  ) |>
  ungroup()
df_sbab_sum$Method1 <- "sensi-boost"
df_sbab_sum$Method2 <- "ablation"



# sensi-boost vs kurt-boost ----------------------------------------

df_sensi_kurt <- df_all |>
  filter(
    attempt != "mxq1" & algo == "mxq"
  ) |>
  filter(
    !grepl("ablation", attempt)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4"
    )
  ) |>
  mutate(
    attempt = sapply(attempt, abbrev_sensi_kurt)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = c(ppl_wikitext, ppl_c4),
    names_vary = "slowest"
  ) |>
  pivot_longer(
    cols = !c("model", "algo", "config", "bpp"),
    names_to = c(".value", "setting"),
    names_pattern = "^(.*)(\\d\\d$)",
  ) |>
  mutate(
    ppl_wikitext_SB = round(ppl_wikitext_SB, digits = 2),
    ppl_c4_SB = round(ppl_c4_SB, digits = 2),
    ppl_wikitext_KB = round(ppl_wikitext_KB, digits = 2),
    ppl_c4_KB = round(ppl_c4_KB, digits = 2)
  ) |>
  mutate(
    score_wk = round(ppl_wikitext_SB - ppl_wikitext_KB, digits = 4),
    score_c4 = round(ppl_c4_SB - ppl_c4_KB, digits = 4)
  ) |>
  mutate(
    wins = ifelse(score_wk < 0, 1, 0) + ifelse(score_c4 < 0, 1, 0),
    ties = ifelse(score_wk == 0, 1, 0) + ifelse(score_c4 == 0, 1, 0),
    losses = ifelse(score_wk > 0, 1, 0) + ifelse(score_c4 > 0, 1, 0)
  ) |>
  select(
    c(
      "model",
      "algo",
      "config",
      "bpp",
      "setting",
      "wins",
      "ties",
      "losses",
      "ppl_wikitext_SB",
      "ppl_wikitext_KB",
      "ppl_c4_SB",
      "ppl_c4_KB",
    )
  )

df_sk_sum <- df_sensi_kurt |>
  group_by(model) |>
  summarise(
    wins = sum(wins),
    ties = sum(ties),
    losses = sum(losses)
  ) |>
  ungroup()
df_sk_sum$Method1 <- "sensi-boost"
df_sk_sum$Method2 <- "kurt-boost"

df_ks_sum <- df_sk_sum |>
  mutate(
    tmp = wins,
    wins = losses,
    losses = tmp
  ) |>
  select(!c("tmp"))

df_ks_sum$Method1 <- "kurt-boost"
df_ks_sum$Method2 <- "sensi-boost"

df_sum <- bind_rows(
  df_sb_sum,
  df_sb_mxq_sum,
  df_kb_sum,
  df_kb_mxq_sum,
  df_sk_sum,
  # df_ks_sum,
  df_sbab_sum,
  df_kbab_sum
) |>
  mutate(
    Method1 = R.utils::toCamelCase(Method1, split = "-", capitalize = TRUE),
    Method2 = R.utils::toCamelCase(Method2, split = "-", capitalize = TRUE),
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
      labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
    ),
  )
write_csv(
  df_sum,
  "df_win_loss.csv"
)


# Calculate percentages and convert to long format
df_disp <- df_sum |>
  mutate(Method1 = sapply(Method1, abbr_method)) |>
  mutate(Method2 = sapply(Method2, abbr_method)) |>
  mutate(Method = paste0(Method1, " vs ", Method2)) |>
  mutate(Total = wins + losses + ties) |>
  mutate(across(c(wins, losses, ties), ~ (.x / Total) * 100)) |>
  tidyr::pivot_longer(
    cols = c(wins, losses, ties),
    names_to = "Result",
    values_to = "Percentage"
  ) |>
  mutate(
    Method = factor(
      Method,
      levels = c(
        "SB vs ABL",
        "SB vs HQQ",
        "SB vs MXQ",
        "SB vs KB",
        "KB vs ABL",
        "KB vs HQQ",
        "KB vs MXQ"
      )
    ),
    Result = factor(
      Result,
      levels = c("losses", "ties", "wins")
    )
  )

# Create the horizontal stacked bar chart
plt <- ggplot(
  # results_long, aes(x = Percentage, y = as.numeric(Method1), fill = Result)
  df_disp, aes(x = Percentage, y = Method, fill = Result)
) +
  geom_col(width = 5.8) +
  # Nature/Science-inspired color palette
  scale_fill_manual(values = c(
    "wins" = "#66c2a5", # Steel blue
    "losses" = "#fc8d62", # Muted red
    "ties" = "#8da0cb" # Purple
  )) +
  # Add percentage labels
  geom_text(
    data = subset(df_disp, Percentage > 0),
    aes(label = sprintf("%.0f%%", Percentage)),
    position = position_stack(vjust = 0.5),
    color = "white",
    size = 4.5
  ) +
  theme_minimal() +
  labs(
    title = "",
    x = NULL,
    y = NULL,
    fill = "Result"
  ) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    plot.title = element_text(
      hjust = 0.5, size = 14, face = "bold", color = "black"
    ),
    legend.position = "left",
    legend.text = element_text(color = "black"),
    strip.text.x = element_text(face = "bold", size = 16),
    strip.text.y = element_text(size = 12, angle = 60, face = "bold"),
    legend.title = element_text(color = "black", face = "bold")
  ) +
  facet_grid(Method ~ model)
ggsave(
  "pdfs/wtl-llama-comparison.pdf",
  plot = plt,
  width = 13,
  height = 7
)
