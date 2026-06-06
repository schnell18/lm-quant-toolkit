#!/usr/bin/env Rscript

library(tidyverse)
library(knitr)
library(kableExtra)
library(optparse)

# Column-group order of the report. Llama-2 sizes grouped, then Llama-3.
REPORT_MODELS <- c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")

shorten_model <- function(model) {
  if (str_detect(model, "Llama-2-7b")) {
    return("Llama-2-7B")
  } else if (str_detect(model, "Llama-2-13b")) {
    return("Llama-2-13B")
  } else if (str_detect(model, "Llama-3-8B")) {
    return("Llama-3-8B")
  }
  return(model)
}

# Within a group, bold the min, underline the 2nd-min (lower PPL is better).
# Ties at the min keep all tied entries bold (rank 1, ties.method = "min").
mark_best_second <- function(x) {
  fmt <- ifelse(is.na(x), NA_character_, sprintf("%.4f", x))
  r <- rank(x, ties.method = "min", na.last = "keep")
  best <- !is.na(r) & r == 1
  second <- !is.na(r) & r == 2
  fmt[best] <- cell_spec(fmt[best], format = "latex", bold = TRUE)
  fmt[second] <- cell_spec(fmt[second], format = "latex", underline = TRUE)
  fmt
}

# Full (method, config) skeleton so all 4 KurtBoost configs always appear;
# absent runs (e.g. KB11/KB21 in round1) render as "-".
build_skeleton <- function(models) {
  base_row <- tibble(method = "FP16", config = "base")
  quant_rows <- crossing(
    method = c("HQQ+", "KB11", "KB12", "KB21", "KB22"),
    config = c("b1g8", "b2g8")
  )
  bind_rows(base_row, quant_rows) |>
    crossing(model = models)
}

dump_latex_table <- function(df, experiment, models,
                             latex_file = "hqqplus-kurtboost.tex") {
  options(knitr.kable.NA = "-")
  # One {WikiText2, MEM} pair per model, after the Method/Config columns.
  header_above <- c(2, rep(2, length(models)))
  names(header_above) <- c(" ", models)
  tabular <- df |>
    kable(
      format = "latex",
      booktabs = TRUE,
      longtable = FALSE,
      linesep = "",
      escape = FALSE,
      align = strrep("c", 2 + 2 * length(models)),
      caption = paste0("Perplexity (WikiText2) and memory of ", experiment),
      label = "tab:hqqplus-kurtboost-result",
      col.names = c(
        "Method", "Config",
        rep(c("WikiText2", "MEM"), length(models))
      )
    ) |>
    kable_styling(
      # latex_options = c("HOLD_position"),
      font_size = 8
    ) |>
    add_header_above(
      header_above,
      include_empty = TRUE,
      line_sep = 0
    ) |>
    collapse_rows(
      columns = 2,
      latex_hline = "major"
    ) |>
    footnote(
      general = paste(
        "Best PPL per config in \\textbf{bold}, second best",
        "\\underline{underlined}. MEM reported in GiB."
      ),
      general_title = "Note:",
      footnote_as_chunk = TRUE,
      escape = FALSE,
      threeparttable = TRUE
    )

  head <- r"(
\documentclass{article}
\usepackage{booktabs,makecell,multirow,threeparttable}
\usepackage{array,caption}

\begin{document}

)"
  tail <- r"(

\end{document}
)"
  out <- paste(head, tabular, tail, sep = "\n")

  dir.create("pdfs", showWarnings = FALSE)
  fh <- file(paste0("pdfs/", latex_file))
  writeLines(out, fh)
  close(fh)
}

process_dataframe <- function(df, models, method_levels, config_levels) {
  ppl_cols <- paste0("ppl_wikitext_", models)
  mem_cols <- paste0("memory_", models)
  # Interleave ppl/mem per model: ppl_m1, mem_m1, ppl_m2, mem_m2, ...
  latex_cols <- c(
    "method", "config",
    as.vector(rbind(ppl_cols, mem_cols))
  )

  df_short <- df |>
    mutate(model = sapply(model, shorten_model)) |>
    filter(model %in% models)

  build_skeleton(models) |>
    left_join(df_short, join_by(method, config, model)) |>
    mutate(
      ppl_wikitext = round(ppl_wikitext, digits = 4),
      method = factor(method, levels = method_levels),
      config = factor(config, levels = config_levels)
    ) |>
    select(model, method, config, ppl_wikitext, memory) |>
    pivot_wider(
      names_from = model,
      values_from = c(ppl_wikitext, memory),
      names_vary = "slowest"
    ) |>
    group_by(config) |>
    mutate(across(all_of(ppl_cols), mark_best_second)) |>
    ungroup() |>
    mutate(across(
      all_of(mem_cols),
      ~ ifelse(is.na(.x), NA_character_, sprintf("%.2f", .x))
    )) |>
    arrange(config, method) |>
    select(all_of(latex_cols))
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--csv_file"),
  type = "character",
  help = "The combined csv file",
  metavar = "character"
)
parser <- add_option(
  parser, c("--caption"),
  type = "character",
  help = "Caption label",
  metavar = "character"
)
args <- parse_args(parser)

csv_fp <- if (is.null(args$csv_file)) "combined.csv" else args$csv_file
the_caption <- if (is.null(args$caption)) "HQQ+ vs. KurtBoost" else args$caption

df_all <- read_csv(csv_fp, show_col_types = FALSE)
method_levels <- c("FP16", "HQQ+", "KB11", "KB12", "KB21", "KB22")
config_levels <- c("base", "b1g8", "b2g8")

df_latex <- process_dataframe(df_all, REPORT_MODELS, method_levels, config_levels)
dump_latex_table(df_latex, the_caption, REPORT_MODELS)
