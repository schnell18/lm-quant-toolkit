#!/usr/bin/env Rscript

library(tidyverse)
library(openxlsx)
library(knitr)
library(kableExtra)
library(optparse)
library(this.path)

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
    return(budget)
  }
}

dump_latex_table <- function(df, experiment, latex_file = "table.tex") {
  options(knitr.kable.NA = "-")
  tabular <- df |>
    kable(
      format = "latex",
      booktabs = TRUE,
      linesep = "",
      escape = FALSE,
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
shorten_attempt <- function(attempt) {
  # Map kurt-boost-X-Y to KBXY; fall back to attempt unchanged otherwise.
  if (is.na(attempt)) {
    return(NA_character_)
  }
  m <- regmatches(
    attempt,
    regexec("^kurt-boost-([0-9]+)-([0-9]+)$", attempt)
  )[[1]]
  if (length(m) == 3) {
    return(paste0("KB", m[2], m[3]))
  }
  return(attempt)
}

mark_best_second <- function(x) {
  # Within a group, bold the min, underline the 2nd-min.
  # Ties at the min keep all tied entries bold (rank 1, ties.method = "min").
  fmt <- ifelse(is.na(x), NA_character_, sprintf("%.2f", x))
  r <- rank(x, ties.method = "min", na.last = "keep")
  best <- !is.na(r) & r == 1
  second <- !is.na(r) & r == 2
  fmt[best] <- cell_spec(fmt[best], format = "latex", bold = TRUE)
  fmt[second] <- cell_spec(fmt[second], format = "latex", underline = TRUE)
  fmt
}

process_dataframe <- function(df, method_levels, method_labels) {
  all_cols <- c(
    "model", "method", "config",
    "bpp", "ppl_wikitext", "ppl_c4",
    "memory"
  )
  latex_cols <- c(
    "method", "config", "bpp",
    "ppl_wikitext_Qwen3.5-2B", "ppl_c4_Qwen3.5-2B", "memory_Qwen3.5-2B",
    "ppl_wikitext_Qwen3.5-4B", "ppl_c4_Qwen3.5-4B", "memory_Qwen3.5-4B",
    "ppl_wikitext_Qwen3.5-9B", "ppl_c4_Qwen3.5-9B", "memory_Qwen3.5-9B"
  )
  ppl_cols <- c(
    "ppl_wikitext_Qwen3.5-2B", "ppl_c4_Qwen3.5-2B",
    "ppl_wikitext_Qwen3.5-4B", "ppl_c4_Qwen3.5-4B",
    "ppl_wikitext_Qwen3.5-9B", "ppl_c4_Qwen3.5-9B"
  )
  mem_cols <- c(
    "memory_Qwen3.5-2B", "memory_Qwen3.5-4B", "memory_Qwen3.5-9B"
  )
  df_latex <- df |>
    mutate(
      method = ifelse(
        algo == "mxq",
        sapply(attempt, shorten_attempt),
        toupper(algo)
      ),
      config = ifelse(algo == "mxq", sapply(bpp, budget_to_cfg), config)
    ) |>
    mutate(
      ppl_wikitext = round(ppl_wikitext, digits = 2),
      ppl_c4 = round(ppl_c4, digits = 2),
      memory = round(load_mem_allot, digits = 2)
    ) |>
    mutate(
      method = factor(
        method,
        levels = method_levels,
        labels = method_labels
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
    filter(!is.na(method), !is.na(config)) |>
    select(all_of(all_cols)) |>
    pivot_wider(
      names_from = model,
      values_from = c(ppl_wikitext, ppl_c4, memory),
      names_vary = "slowest"
    ) |>
    group_by(config) |>
    mutate(across(all_of(ppl_cols), mark_best_second)) |>
    ungroup() |>
    mutate(across(all_of(mem_cols),
      ~ ifelse(is.na(.x), NA_character_, sprintf("%.2f", .x)))) |>
    select(all_of(latex_cols)) |>
    arrange(config, method, desc(bpp))

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
method_levels <- c(
  "FP16", "HQQ", "AWQ", "GPTQ", "BNB",
  "KB20", "KB21", "KB22", "KB23",
  "KB30", "KB31", "KB32", "KB33"
)
method_labels <- c(
  "FP16", "HQQ", "AWQ", "GPTQ", "BnB",
  "KB20", "KB21", "KB22", "KB23",
  "KB30", "KB31", "KB32", "KB33"
)
df_latex <- process_dataframe(df_all, method_levels, method_labels)
dump_latex_table(df_latex, the_attempt)
