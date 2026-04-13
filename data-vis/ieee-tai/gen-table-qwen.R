#!/usr/bin/env Rscript

library(tidyverse)
library(openxlsx)
library(knitr)
library(kableExtra)
library(optparse)
library(this.path)

dump_latex_table <- function(df, experiment, latex_file = "table.tex") {
  options(knitr.kable.NA = "-")
  tabular <- df |>
    kable(
      format = "latex",
      booktabs = TRUE,
      linesep = "",
      align = c("cccccccccccc"),
      caption = paste0("PPL results of ", experiment),
      label = "tab:experiment-result",
      col.names = c(
        "Method", "Config", "BPP",
        "WikiText2", "C4", "MEM",
        "WikiText2", "C4", "MEM",
        "WikiText2", "C4", "MEM"
      )
    ) |>
    kable_styling(latex_options = c("hold_position")) |>
    add_header_above(
      c(" " = 3, "Qwen3.5-2B" = 3, "Qwen3.5-4B" = 3, "Qwen3.5-9B" = 3)
    ) |>
    collapse_rows(columns = 2, latex_hline = "major")

  tabular <- gsub(
    "\\begin{tabular}",
    "\\begin{adjustbox}{width=\\textwidth,keepaspectratio}\n\\begin{tabular}",
    tabular,
    fixed = TRUE
  )
  tabular <- gsub(
    "\\end{tabular}",
    "\\end{tabular}\n\\end{adjustbox}",
    tabular,
    fixed = TRUE
  )

  head <- r"(
\documentclass{article}
\usepackage{booktabs,makecell,multirow,threeparttable}
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

  fh <- file(paste0("pdfs/", latex_file))
  writeLines(out, fh)
  close(fh)
}
process_dataframe <- function(df, algo_levels, algo_labels) {
  all_cols <- c(
    "model", "algo", "config",
    "bpp", "ppl_wikitext", "ppl_c4",
    "memory"
  )
  latex_cols <- c(
    "algo", "config", "bpp",
    "ppl_wikitext_Qwen3.5-2B", "ppl_c4_Qwen3.5-2B", "memory_Qwen3.5-2B",
    "ppl_wikitext_Qwen3.5-4B", "ppl_c4_Qwen3.5-4B", "memory_Qwen3.5-4B",
    "ppl_wikitext_Qwen3.5-9B", "ppl_c4_Qwen3.5-9B", "memory_Qwen3.5-9B"
  )
  df_latex <- df |>
    mutate(
      config = ifelse(algo == "mxq", sapply(bpp, budget_to_cfg), config)
    ) |>
    mutate(
      ppl_wikitext = round(ppl_wikitext, digits = 2),
      ppl_c4 = round(ppl_c4, digits = 2),
      memory = round(load_mem_allot, digits = 2)
    ) |>
    mutate(
      algo = factor(
        algo,
        levels = algo_levels,
        labels = algo_labels
      ),
      config = factor(
        config,
        levels = c(
          "base",
          "b8g32", "b8g64", "b8g128",
          "b4g32", "b4g64", "b4g128",
          "b3g32", "b3g64", "b3g128"
        )
      )
    ) |>
    select(all_of(all_cols)) |>
    pivot_wider(
      names_from = model,
      values_from = c(ppl_wikitext, ppl_c4, memory),
      names_vary = "slowest"
    ) |>
    select(all_of(latex_cols)) |>
    arrange(config, algo, desc(bpp))

  return(df_latex)
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--csv_file"),
  type = "character",
  help = "The combined csv file",
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
  the_attempt <- "baseline"
} else {
  the_attempt <- args$attempt
}

df_all <- read_csv(csv_fp)
# levels <- c("mxq", "hqq", "fp16", "awq", "gptq", "bnb")
# labels <- c("MXQ", "HQQ", "FP16", "AWQ", "GPTQ", "BnB")
levels <- c("fp16", "awq", "gptq", "bnb")
labels <- c("FP16", "AWQ", "GPTQ", "BnB")
df_latex <- process_dataframe(df_all, levels, labels)
dump_latex_table(df_latex, the_attempt)
