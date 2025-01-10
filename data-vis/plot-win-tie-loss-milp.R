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
  if (method == "SensiMilp") {
    return("SM")
  } else if (method == "KurtMilp") {
    return("KM")
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
    algo == "mxq" & attempt == "mxq2"
  )

# sensi-milp vs HQQ ------------------------------------------------------
sidea_name <- "sensi-milp"
sideb_name <- "HQQ"
df_side_a <- df_all |>
  filter(grepl(sidea_name, attempt)) |>
  filter(!grepl("-abl", attempt)) |>
  filter(bpp == 4.51 | bpp == 4.25 | bpp == 4.13 |
    bpp == 3.51 | bpp == 3.25 | bpp == 3.13)
df_sm_hqq_sum <- calc_wtl_score(
  df_side_a,
  df_hqq,
  sidea_name,
  sideb_name
)

# sensi-milp vs MXQ ------------------------------------------------------
sidea_name <- "sensi-milp"
sideb_name <- "MXQ"
df_side_a <- df_all |>
  filter(grepl(sidea_name, attempt)) |>
  filter(!grepl("-abl", attempt))
df_sm_mxq_sum <- calc_wtl_score(
  df_side_a,
  df_mxq,
  sidea_name,
  sideb_name,
  dump_det_data = TRUE
)

# kurt-milp vs MXQ ------------------------------------------------------
sidea_name <- "kurt-milp"
sideb_name <- "MXQ"
df_side_a <- df_all |>
  filter(grepl(sidea_name, attempt)) |>
  filter(!grepl("-abl", attempt))
df_km_mxq_sum <- calc_wtl_score(
  df_side_a,
  df_mxq,
  sidea_name,
  sideb_name,
  dump_det_data = TRUE
)

# kurt-milp vs HQQ ------------------------------------------------------
sidea_name <- "kurt-milp"
sideb_name <- "HQQ"
df_side_a <- df_all |>
  filter(grepl(sidea_name, attempt)) |>
  filter(!grepl("-abl", attempt)) |>
  filter(bpp == 4.51 | bpp == 4.25 | bpp == 4.13 |
    bpp == 3.51 | bpp == 3.25 | bpp == 3.13)
df_km_hqq_sum <- calc_wtl_score(
  df_side_a,
  df_hqq,
  sidea_name,
  sideb_name
)

# kurt-milp vs kurt-milp ablation ----------------------------------------
sidea_name <- "kurt-milp"
sideb_name <- "ablation"
df_side_a <- pick_attempt_data_milp(
  df_all |> filter(!grepl("kurt-milp-abl", attempt)),
  sidea_name
)
df_side_b <- df_all |> filter(grepl("kurt-milp-abl", attempt))

df_kmab_sum <- calc_wtl_score(
  df_side_a,
  df_side_b,
  sidea_name,
  sideb_name,
  c("model", "bpp")
)

# sensi-milp vs sensi-milp ablation ----------------------------------------
sidea_name <- "sensi-milp"
sideb_name <- "ablation"
df_side_a <- pick_attempt_data_milp(
  df_all |> filter(!grepl("sensi-milp-abl", attempt)),
  sidea_name
)
df_side_b <- df_all |> filter(grepl("sensi-milp-abl", attempt))
df_smab_sum <- calc_wtl_score(
  df_side_a,
  df_side_b,
  sidea_name,
  sideb_name,
  c("model", "bpp")
)

# sensi-milp vs kurt-milp ----------------------------------------
sidea_name <- "sensi-milp"
sideb_name <- "kurt-milp"
df_side_a <- pick_attempt_data_milp(
  df_all |> filter(!grepl("sensi-milp-abl", attempt)),
  sidea_name
)
df_side_b <- pick_attempt_data_milp(
  df_all |> filter(!grepl("kurt-milp-abl", attempt)),
  sideb_name
)

df_sk_sum <- calc_wtl_score(
  df_side_a,
  df_side_b,
  sidea_name,
  sideb_name,
  c("model", "bpp", "setting")
)

df_sum <- bind_rows(
  df_sm_hqq_sum,
  df_sm_mxq_sum,
  df_km_hqq_sum,
  df_km_mxq_sum,
  df_sk_sum,
  df_smab_sum,
  df_kmab_sum
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
        "SM vs ABL",
        "SM vs HQQ",
        "SM vs MXQ",
        "SM vs KM",
        "KM vs ABL",
        "KM vs HQQ",
        "KM vs MXQ"
      )
    ),
    Result = factor(
      Result,
      levels = c("losses", "ties", "wins")
    )
  )

plot_win_tie_loss(df_disp, "wtl-milp-comparison.pdf")
