pick_attempt_data <- function(df, attempt_prefix) {
  pattern_vector <- c("\\w+-\\w+", "-", stop = "\\d", "-", topm = "\\d")
  df_side <- df |>
    filter(grepl(attempt_prefix, attempt)) |>
    mutate(atmpt = attempt) |>
    separate_wider_regex(
      atmpt,
      pattern_vector
    ) |>
    mutate(
      setting = paste0(stop, topm)
    ) |>
    select(!c("stop", "topm"))
  return(df_side)
}

pick_attempt_data_milp <- function(df, attempt_prefix) {
  df_side <- df |>
    filter(grepl(attempt_prefix, attempt)) |>
    mutate(atmpt = attempt) |>
    separate_wider_regex(
      atmpt,
      c("(?:\\w+-)+", topm = "\\d")
    ) |>
    mutate(
      setting = topm
    ) |>
    select(!c("topm"))
  return(df_side)
}

# Mandatory columns df_side_x:
#   model, algo, config, bpp, attempt, ppl_wikitext, ppl_c4, load_mem_allot
# Optional columns df_side_x: setting
calc_wtl_score_det <- function(
    df_side_a,
    df_side_b,
    sidea_name,
    sideb_name,
    join_cond = c("model", "bpp"),
    dump_det_data = FALSE) {
  key_fields <- c(
    "algo",
    "config",
    "attempt"
  )
  value_fields <- c(
    "ppl_wikitext",
    "ppl_c4"
  )
  key_fields <- c(key_fields, join_cond)
  fields <- c(key_fields, value_fields)

  df_det <- df_side_a |>
    select(
      all_of(fields)
    ) |>
    left_join(
      df_side_b,
      suffix = c("_sidea", ""),
      by = join_cond
    ) |>
    mutate(
      ppl_wikitext_sidea = round(ppl_wikitext_sidea, digits = 2),
      ppl_c4_sidea = round(ppl_c4_sidea, digits = 2),
      ppl_wikitext = round(ppl_wikitext, digits = 2),
      ppl_c4 = round(ppl_c4, digits = 2)
    ) |>
    mutate(
      score_wk = round(ppl_wikitext_sidea - ppl_wikitext, digits = 4),
      score_c4 = round(ppl_c4_sidea - ppl_c4, digits = 4)
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
        "attempt_sidea",
        "wins",
        "losses",
        "ties",
        "ppl_wikitext_sidea",
        "ppl_wikitext",
        "ppl_c4_sidea",
        "ppl_c4"
      )
    )
  if (dump_det_data) {
    write.xlsx(
      df_det,
      paste0("df_det-", sidea_name, "-", sideb_name, ".xlsx"),
      asTable = TRUE
    )
  }
  return(df_det)
}

# Mandatory columns df_side_x:
#   model, algo, config, bpp, attempt, ppl_wikitext, ppl_c4, load_mem_allot
# Optional columns df_side_x: setting
calc_wtl_score <- function(
    df_side_a,
    df_side_b,
    sidea_name,
    sideb_name,
    join_cond = c("model", "bpp"),
    dump_det_data = FALSE) {
  df_det <- calc_wtl_score_det(
    df_side_a,
    df_side_b,
    sidea_name,
    sideb_name,
    join_cond = join_cond,
    dump_det_data = dump_det_data
  )
  df_sum <- df_det |>
    group_by(model) |>
    summarise(
      wins = sum(wins),
      ties = sum(ties),
      losses = sum(losses)
    ) |>
    ungroup()
  df_sum$Method1 <- sidea_name
  df_sum$Method2 <- sideb_name

  return(df_sum)
}
# Create the horizontal stacked bar chart
plot_win_tie_loss <- function(df_disp, output_file) {
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
      aes(label = sprintf("%.0f", Percentage)),
      position = position_stack(vjust = 0.5),
      color = "white",
      size = 6.0
    ) +
    theme_minimal() +
    guides(fill = guide_legend(reverse = TRUE)) +
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
      legend.text = element_text(color = "black", size = 26),
      legend.title = element_blank(),
      strip.text.x = element_text(face = "bold", size = 26),
      strip.text.y = element_text(size = 14, angle = 40, face = "bold", hjust = 0.01, vjust = -0.01)
    ) +
    facet_grid(Method ~ model)
  ggsave(
    paste0("pdfs/", output_file),
    plot = plt,
    width = 13,
    height = 7
  )
}
