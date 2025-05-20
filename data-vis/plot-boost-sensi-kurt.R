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
source(file.path(here("functions"), "allocation.R"))

load_ppl_mem_inc <- function(allot_cfg_csv_fp, combined_csv_fp) {
  df_cfgs <- read_csv(allot_cfg_csv_fp) |>
    mutate(
      model = factor(
        model,
        levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
        labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
      )
    )

  df_w_base <- read_csv(combined_csv_fp) |>
    mutate(
      model = factor(
        model,
        levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
        labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
      )
    )

  df_hqq <- df_w_base |>
    filter(
      algo == "hqq"
    )

  df_cfg_mem <- calc_mem_inc(df_cfgs)

  df_ppl_mem_inc <- df_cfg_mem |>
    left_join(
      df_hqq,
      suffix = c("", "_h"),
      by = join_by(model, bit_budget == bpp)
    ) |>
    left_join(
      df_w_base,
      suffix = c("_hqq", ""),
      by = join_by(model, attempt, bit_budget == bpp)
    ) |>
    mutate(
      ppl_wikitext_decr = round(
        100 * (ppl_wikitext_hqq - ppl_wikitext) / ppl_wikitext_hqq,
        digits = 2
      ),
      ppl_c4_decr = round(
        100 * (ppl_c4_hqq - ppl_c4) / ppl_c4_hqq,
        digits = 2
      ),
      mem_incr = round(
        100 * (load_mem_allot - load_mem_allot_hqq) / load_mem_allot_hqq,
        digits = 2
      )
    ) |>
    rename(bpp = bit_budget) |>
    select(
      c(
        "model",
        "attempt",
        "bpp",
        "increment",
        "ppl_wikitext_decr",
        "ppl_c4_decr",
        "mem_incr",
        "ppl_wikitext",
        "ppl_c4",
        "ppl_wikitext_hqq",
        "ppl_c4_hqq",
        "mem_orig",
        "mem_new",
        "load_mem_allot",
        "load_mem_allot_hqq"
      )
    )

  df_ppl_mem_inc <- df_ppl_mem_inc |>
    filter(
      !is.na(attempt) & attempt != "mxq1"
    ) |>
    filter(
      !grepl("-abl", attempt)
    ) |>
    separate_wider_regex(
      attempt,
      c(method = "\\w+-\\w+", "-", stop_topm = "\\d-\\d")
    ) |>
    mutate(
      method1 = ifelse(grepl("sensi-", method), "SensiBoost", "KurtBoost"),
      wikitext_decr = ppl_wikitext_decr,
      c4_decr = ppl_c4_decr
    ) |>
    select(!c("method1", "ppl_wikitext_decr", "ppl_c4_decr")) |>
    pivot_longer(
      cols = c("wikitext_decr", "c4_decr"),
      names_to = c("dataset", ".value"),
      names_sep = "_"
    ) |>
    mutate(
      method = factor(
        method,
        levels = c("kurt-boost", "sensi-boost"),
        labels = c("KurtBoost", "SensiBoost")
      ),
      dataset = factor(
        dataset,
        levels = c("wikitext", "c4"),
        labels = c("WikiText2", "C4")
      )
    )
  return(df_ppl_mem_inc)
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

# TODO: remove the debug line
csv_file <- "endeavors/boost/data/combined.csv"
allot_csv_file <- "endeavors/boost/data/quant-cfg-allocation.csv"

df_disp <- load_ppl_mem_inc(allot_csv_file, csv_file)

plt <- ggplot(
  df_disp, aes(x = increment, y = decr, shape = method, color = dataset)
) +
  geom_point(size = 1.5) +
  labs(x = "% Memory Increment", y = "% Perplexity Drop") +
  theme(
    strip.background = element_rect(
      color = "darkgray", fill = "white", linewidth = 1.0, linetype = "solid"
    ),
    strip.text.x = element_text(face = "bold", size = 12),
    strip.text.y = element_text(face = "bold", size = 12),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    legend.position = "bottom"
  ) +
  guides(
    color = guide_legend(title = "Dataset"),
    shape = guide_legend(title = "Method")
  ) +
  scale_y_continuous(labels = function(y) format(y, nsmall = 2, scientific = FALSE)) +
  facet_grid(model ~ bpp, scales = "free")

ggsave(
  "pdfs/ppl-decr-mem-inc.pdf",
  plot = plt,
  width = 8,
  height = 5,
  dpi = 600
)
