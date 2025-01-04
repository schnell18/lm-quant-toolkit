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
      c(" " = 3, "Llama-2-7B" = 3, "Llama-2-13B" = 3, "Llama-3-8B" = 3)
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
    "ppl_wikitext_Llama-2-7B", "ppl_c4_Llama-2-7B", "memory_Llama-2-7B",
    "ppl_wikitext_Llama-2-13B", "ppl_c4_Llama-2-13B", "memory_Llama-2-13B",
    "ppl_wikitext_Llama-3-8B", "ppl_c4_Llama-3-8B", "memory_Llama-3-8B"
  )
  df_latex <- df |>
    mutate(
      config = ifelse(algo == "mxq", sapply(bpp, budget_to_cfg), config)
    ) |>
    mutate(
      attempt = ifelse(
        is.na(attempt),
        attempt,
        sapply(attempt, abbrev_sensi_kurt)
      )
    ) |>
    mutate(
      algo = ifelse(
        algo == "mxq" & !is.na(attempt) & attempt != "mxq1" & attempt != "mxq2",
        paste0(algo, "-", attempt),
        algo
      ),
      ppl_wikitext = round(ppl_wikitext, digits = 2),
      ppl_c4 = round(ppl_c4, digits = 2),
      memory = round(load_mem_allot, digits = 2)
    ) |>
    mutate(
      model = factor(
        model,
        levels = c("Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B"),
        labels = c("Llama-2-7B", "Llama-2-13B", "Llama-3-8B")
      ),
      algo = factor(
        algo,
        levels = algo_levels,
        labels = algo_labels
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
    select(all_of(all_cols)) |>
    filter(bpp >= 3.03 & bpp < 8.0 | bpp >= 16.00) |>
    pivot_wider(
      names_from = model,
      values_from = c(ppl_wikitext, ppl_c4, memory),
      names_vary = "slowest"
    ) |>
    select(all_of(latex_cols)) |>
    arrange(config, algo, desc(bpp))

  return(df_latex)
}
