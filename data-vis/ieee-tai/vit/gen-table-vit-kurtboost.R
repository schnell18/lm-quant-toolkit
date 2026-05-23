#!/usr/bin/env Rscript

library(tidyverse)
library(knitr)
library(kableExtra)
library(optparse)

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
  } else if (budget == 3.02) {
    return("b2g16")
  } else {
    return(as.character(budget))
  }
}

shorten_attempt <- function(attempt) {
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

shorten_model <- function(model) {
  if (str_detect(model, "ViT-B-32")) {
    return("B-32")
  } else if (str_detect(model, "ViT-L-14")) {
    return("L-14")
  } else if (str_detect(model, "ViT-H-14")) {
    return("H-14")
  }
  return(model)
}

# Bold the max, underline the second-max within a group (higher is better).
mark_best_second_max <- function(x) {
  fmt <- ifelse(is.na(x), NA_character_, sprintf("%.2f", x))
  r <- rank(-x, ties.method = "min", na.last = "keep")
  best <- !is.na(r) & r == 1
  second <- !is.na(r) & r == 2
  fmt[best] <- cell_spec(fmt[best], format = "latex", bold = TRUE)
  fmt[second] <- cell_spec(fmt[second], format = "latex", underline = TRUE)
  fmt
}

dump_latex_table <- function(df, experiment, latex_file = "vit-kurtboost.tex") {
  options(knitr.kable.NA = "-")
  tabular <- df |>
    kable(
      format = "latex",
      booktabs = TRUE,
      longtable = TRUE,
      linesep = "",
      escape = FALSE,
      align = c("cccccccccccc"),
      caption = paste0("Zero-shot results of ", experiment),
      label = "tab:vit-kurtboost-result",
      col.names = c(
        "Method", "Config", "BPP",
        "Acc@1", "Acc@5", "MEM",
        "Acc@1", "Acc@5", "MEM",
        "Acc@1", "Acc@5", "MEM"
      )
    ) |>
    kable_styling(
      latex_options = c("repeat_header"),
      font_size = 8,
      repeat_header_method = "replace",
      repeat_header_continued = "\\textit{(continued on next page)}"
    ) |>
    add_header_above(
      c(" " = 3, "CLIP-ViT-B-32" = 3, "CLIP-ViT-L-14" = 3, "CLIP-ViT-H-14" = 3),
      include_empty = TRUE,
      line_sep = 0
    ) |>
    collapse_rows(
      columns = 2,
      latex_hline = "major",
      longtable_clean_cut = TRUE
    )

  head <- r"(
\documentclass{article}
\usepackage{booktabs,makecell,multirow,threeparttable}
\usepackage{longtable,array,caption}

\begin{document}

)"
  tail <- r"(

\end{document}
)"
  out <- paste(head, tabular, tail, sep = "\n")

  fh <- file(paste0("pdfs/", latex_file))
  writeLines(out, fh)
  close(fh)
}

process_dataframe <- function(df, method_levels, method_labels) {
  all_cols <- c(
    "model", "method", "config", "bpp",
    "acc1", "acc5", "memory"
  )
  acc_cols <- c(
    "acc1_B-32", "acc5_B-32",
    "acc1_L-14", "acc5_L-14",
    "acc1_H-14", "acc5_H-14"
  )
  mem_cols <- c("memory_B-32", "memory_L-14", "memory_H-14")
  latex_cols <- c(
    "method", "config", "bpp",
    "acc1_B-32", "acc5_B-32", "memory_B-32",
    "acc1_L-14", "acc5_L-14", "memory_L-14",
    "acc1_H-14", "acc5_H-14", "memory_H-14"
  )

  df_latex <- df |>
    mutate(
      model = factor(
        sapply(model, shorten_model),
        levels = c("B-32", "L-14", "H-14")
      ),
      method = ifelse(
        algo == "mxq",
        sapply(attempt, shorten_attempt),
        toupper(algo)
      ),
      config = ifelse(algo == "mxq", sapply(bpp, budget_to_cfg), config)
    ) |>
    mutate(
      acc1 = round(acc1_zeroshot_cls * 100, digits = 2),
      acc5 = round(acc5_zeroshot_cls * 100, digits = 2),
      memory = round(zeroshot_mem_allot, digits = 2)
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
          "b3g32", "b3g64", "b3g128",
          "b2g16", "b2g32", "b2g64", "b2g128"
        )
      )
    ) |>
    filter(!is.na(method), !is.na(config), !is.na(model)) |>
    select(all_of(all_cols)) |>
    pivot_wider(
      names_from = model,
      values_from = c(acc1, acc5, memory),
      names_vary = "slowest"
    ) |>
    group_by(config) |>
    mutate(across(all_of(acc_cols), mark_best_second_max)) |>
    ungroup() |>
    mutate(across(
      all_of(mem_cols),
      ~ ifelse(is.na(.x), NA_character_, sprintf("%.2f", .x))
    )) |>
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
  help = "Caption label",
  metavar = "character"
)

args <- parse_args(parser)
if (is.null(args$csv_file)) {
  csv_fp <- "combined-kurtboost.csv"
} else {
  csv_fp <- args$csv_file
}
if (is.null(args$attempt)) {
  the_attempt <- "CLIP/ViT KurtBoost"
} else {
  the_attempt <- args$attempt
}

df_all <- read_csv(csv_fp, show_col_types = FALSE)
method_levels <- c(
  "FP16", "HQQ",
  "KB11", "KB12", "KB13",
  "KB21", "KB22", "KB23"
)
method_labels <- c(
  "FP16", "HQQ",
  "KB11", "KB12", "KB13",
  "KB21", "KB22", "KB23"
)
df_latex <- process_dataframe(df_all, method_levels, method_labels)
dump_latex_table(df_latex, the_attempt)
