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
source(file.path(here("functions"), "wtl.R"))


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
sidea_name <- "sensi-boost"
sideb_name <- "HQQ"

df_side_a <- df_all |> filter(grepl(sidea_name, attempt))
df_sb_sum <- calc_wtl_score(
  df_side_a,
  df_hqq,
  sidea_name,
  sideb_name,
  dump_det_data = TRUE
)

# sensi-boost vs MXQ ------------------------------------------------------
sidea_name <- "sensi-boost"
sideb_name <- "MXQ"

df_side_a <- df_all |> filter(grepl(sidea_name, attempt))
df_sb_mxq_sum <- calc_wtl_score(
  df_side_a,
  df_mxq,
  sidea_name,
  sideb_name
)

# kurt-boost vs MXQ ------------------------------------------------------
sidea_name <- "kurt-boost"
sideb_name <- "MXQ"

df_kb_mxq_sum <- calc_wtl_score(
  df_all |> filter(grepl(sidea_name, attempt)),
  df_mxq,
  sidea_name,
  sideb_name
)

# kurt-boost vs HQQ ------------------------------------------------------
sidea_name <- "kurt-boost"
sideb_name <- "HQQ"

df_kb_sum <- calc_wtl_score(
  df_all |> filter(grepl(sidea_name, attempt)),
  df_hqq,
  sidea_name,
  sideb_name,
  dump_det_data = TRUE
)

# kurt-boost vs kurt-boost ablation ----------------------------------------
sidea_name <- "kurt-boost"
sideb_name <- "ablation"
df_side_a <- pick_attempt_data(df_all, sidea_name)
df_side_b <- pick_attempt_data(df_all, "kurt-abl")

df_kbab_sum <- calc_wtl_score(
  df_side_a,
  df_side_b,
  sidea_name,
  sideb_name,
  c("model", "bpp", "setting")
)

# sensi-boost vs sensi-boost ablation ----------------------------------------
sidea_name <- "sensi-boost"
sideb_name <- "ablation"
df_side_a <- pick_attempt_data(df_all, sidea_name)
df_side_b <- pick_attempt_data(df_all, "sensi-abl")

df_sbab_sum <- calc_wtl_score(
  df_side_a,
  df_side_b,
  sidea_name,
  sideb_name,
  c("model", "bpp", "setting")
)

# sensi-boost vs kurt-boost ----------------------------------------
sidea_name <- "sensi-boost"
sideb_name <- "kurt-boost"
df_side_a <- pick_attempt_data(df_all, sidea_name)
df_side_b <- pick_attempt_data(df_all, sideb_name)

df_sk_sum <- calc_wtl_score(
  df_side_a,
  df_side_b,
  sidea_name,
  sideb_name,
  c("model", "bpp", "setting"),
  dump_det_data = TRUE
)

# df_ks_sum <- df_sk_sum |>
#   mutate(
#     tmp = wins,
#     wins = losses,
#     losses = tmp
#   ) |>
#   select(!c("tmp"))
# df_ks_sum$Method1 <- "kurt-boost"
# df_ks_sum$Method2 <- "sensi-boost"

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
        "KB vs ABL",
        "SB vs HQQ",
        "KB vs HQQ",
        "SB vs MXQ",
        "KB vs MXQ",
        "SB vs KB"
      )
    ),
    Result = factor(
      Result,
      levels = c("losses", "ties", "wins")
    )
  )

plot_win_tie_loss(df_disp, "wtl-boost-comparison.pdf")
