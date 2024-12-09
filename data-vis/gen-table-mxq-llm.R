#!/usr/bin/env Rscript

library(tidyverse)
library(openxlsx)
library(knitr)
library(kableExtra)

budget_to_cfg <- function(budget) {
  if (budget == 3.13) {
    return("b3g128")
  } else if (budget == 3.25) {
    return("b3g64")
  } else if (budget == 3.51) {
    return("b3g32")
  } else if (budget == 4.13) {
    return("b4g128")
  } else if (budget == 4.25) {
    return("b4g64")
  } else if (budget == 4.51) {
    return("b4g32")
  } else if (budget == 8.13) {
    return("b8g128")
  } else if (budget == 8.25) {
    return("b8g64")
  } else if (budget == 8.51) {
    return("b8g32")
  } else if (budget == 2.13) {
    return("b2g128")
  } else if (budget == 2.25) {
    return("b2g64")
  } else if (budget == 2.51) {
    return("b2g32")
  } else {
    return(NA)
  }
}

dump_latex_table <- function(df, latex_file = "table.tex") {
  options(knitr.kable.NA = "-")
  tabular <- df |>
    kable(
      format = "latex",
      booktabs = TRUE,
      longtable = TRUE,
      linesep = "",
      align = c("cccccccccccc"),
      caption = "My formatted LLM quantization table.",
      label = "tab:experiment-result",
      col.names = c(
        "Method", "Config", "BPP",
        "WikiText2", "C4", "MEM",
        "WikiText2", "C4", "MEM",
        "WikiText2", "C4", "MEM"
      )
    ) |>
    kable_styling(
      latex_options = c("hold_position", "repeat_header")
    ) |>
    add_header_above(
      c(" " = 3, "Llama-2-7B" = 3, "Llama-2-13B" = 3, "Llama-3-8B" = 3)
    ) |>
    collapse_rows(columns = 2, latex_hline = "major")

  # tabular <- gsub(
  #   "\\begin{longtable}",
  #   "\\begin{adjustbox}{width=\\textwidth,keepaspectratio}\n\\begin{longtable}",
  #   tabular,
  #   fixed = TRUE
  # )
  # tabular <- gsub(
  #   "\\end{longtable}",
  #   "\\end{longtable}\n\\end{adjustbox}",
  #   tabular,
  #   fixed = TRUE
  # )

  head <- r"(
\documentclass{article}
\usepackage{booktabs,makecell,multirow,threeparttable}
\usepackage{longtable}
\usepackage{adjustbox}

\begin{document}

)"
  tail <- r"(

\end{document}
)"
  out <- paste(
    head,
    tabular,
    tail,
    sep = "\n"
  )

  fh <- file(latex_file)
  writeLines(out, fh)
  close(fh)
}


args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  csv_fp <- "data/combined.csv"
} else {
  csv_fp <- args[1]
}

all_cols <- c(
  "model", "algo", "config",
  "bpp", "ppl_wikitext", "ppl_c4",
  "memory"
)

df_all <- read_csv(csv_fp) |>
  filter(
    is.na(attempt) | attempt == "mxq1"
  ) |>
  mutate(
    ppl_wikitext = round(ppl_wikitext, digits = 2),
    ppl_c4 = round(ppl_c4, digits = 2),
    memory = round(load_mem_allot, digits = 2)
  ) |>
  mutate(
    config = ifelse(algo == "mxq", sapply(bpp, budget_to_cfg), config)
  ) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
      labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
    ),
    algo = factor(
      algo,
      levels = c("mxq", "hqq", "fp16", "awq", "gptq", "bnb"),
      labels = c("MXQ", "HQQ", "FP16", "AWQ", "GPTQ", "BnB"),
    ),
    config = factor(
      config,
      levels = c(
        "base",
        "b4g32", "b4g64", "b4g128",
        "b3g32", "b3g64", "b3g128"
        # "4_51", "4_25", "4_13",
        # "3_51", "3_25", "3_13",
        # "3_65", "6_89", "5_72",
        # "3_07", "4_01", "5_02",
      )
    )
  ) |>
  select(all_of(all_cols))


latex_cols <- c(
  "algo", "config", "bpp",
  "ppl_wikitext_Llama-2-7B", "ppl_c4_Llama-2-7B", "memory_Llama-2-7B",
  "ppl_wikitext_Llama-2-13B", "ppl_c4_Llama-2-13B", "memory_Llama-2-13B",
  "ppl_wikitext_Llama-3-8B", "ppl_c4_Llama-3-8B", "memory_Llama-3-8B"
)

df_latex <- df_all |>
  filter(bpp >= 3.03 & bpp < 8.0 | bpp >= 16.00) |>
  pivot_wider(
    names_from = model,
    values_from = c(ppl_wikitext, ppl_c4, memory),
    names_vary = "slowest"
  ) |>
  select(all_of(latex_cols)) |>
  arrange(config)

dump_latex_table(df_latex)
