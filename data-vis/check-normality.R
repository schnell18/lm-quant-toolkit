#!/usr/bin/env Rscript

library(readr)
library(tidyverse)
library(nortest)

# Function to trim the 5% smallest and biggest values from a vector
trim_extreme_values <- function(x, trim_percent = 0.10) {
  n <- length(x)
  if (n <= 2) {
    return(x)
  } # Can't trim if there are too few values

  # Calculate how many values to trim from each end
  trim_count <- floor(n * trim_percent)

  # If trim_count is 0 (due to small n), make it at least 1
  trim_count <- max(trim_count, 1)

  # Sort and trim
  sorted_x <- sort(x)
  trimmed_x <- sorted_x[(trim_count + 1):(n - trim_count)]

  return(trimmed_x)
}


models <- c("Llama-2-7b-hf", "Meta-Llama-3-8B", "Llama-2-13b-hf")
for (model in models) {
  df_fnorm <- read_csv(
    paste0("~/work/llm-quant/lm-quant-toolkit/src/data/fnorm-", model, ".csv")
  )

  # Calculate the difference in kurtosis between adjacent layers for each module
  df_diff <- df_fnorm |>
    group_by(module, layer) |>
    summarise(
      sensi_score = min(sensitivity),
      kurt_score = min(kurtosis)
    ) |>
    group_by(module) |>
    arrange(module, layer) |>
    mutate(
      kurt_diff = kurt_score - lag(kurt_score),
      sensi_diff = sensi_score / lag(sensi_score)
    ) |>
    filter(!is.na(kurt_diff)) |>
    filter(!is.na(sensi_diff))

  kurt_trim_pct <- 0.10
  sensi_trim_pct <- 0.20
  # Apply trimming for each module and perform Shapiro-Wilk test
  trimmed_results <- df_diff |>
    group_by(module) |>
    summarize(

      # Create a trimmed version of kurt_diff
      kurt_diff_trimmed = list(
        trim_extreme_values(kurt_diff, trim_percent = kurt_trim_pct)
      ),
      kurt_n_trimmed = sapply(kurt_diff_trimmed, length),

      # For Shapiro-Wilk test on trimmed data
      kurt_shapiro_stat = sapply(kurt_diff_trimmed, function(x) {
        if (length(x) >= 3) shapiro.test(x)$statistic else NA
      }),
      kurt_shapiro_p = sapply(kurt_diff_trimmed, function(x) {
        if (length(x) >= 3) shapiro.test(x)$p.value else NA
      }),
      kurt_normal = sapply(kurt_diff_trimmed, function(x) {
        if (length(x) >= 3) shapiro.test(x)$p.value > 0.05 else NA
      }),

      # Create a trimmed version of sensi_diff
      sensi_diff_trimmed = list(
        trim_extreme_values(sensi_diff, trim_percent = sensi_trim_pct)
      ),
      sensi_n_trimmed = sapply(sensi_diff_trimmed, length),

      # For Shapiro-Wilk test on trimmed data
      sensi_shapiro_stat = sapply(sensi_diff_trimmed, function(x) {
        if (length(x) >= 3) shapiro.test(x)$statistic else NA
      }),
      sensi_shapiro_p = sapply(sensi_diff_trimmed, function(x) {
        if (length(x) >= 3) shapiro.test(x)$p.value else NA
      }),
      sensi_normal = sapply(sensi_diff_trimmed, function(x) {
        if (length(x) >= 3) shapiro.test(x)$p.value > 0.05 else NA
      }),
    )

  # Create QQ plots for each module using the trimmed data
  # Unpack the trimmed data for plotting
  df_plot_trimmed_kurt <- df_diff |>
    group_by(module) |>
    do({
      kurt_trimmed_values <- trim_extreme_values(
        .$kurt_diff,
        trim_percent = kurt_trim_pct
      )
      data.frame(
        module = .$module[1],
        kurt_diff_trimmed = kurt_trimmed_values
      )
    })

  df_plot_trimmed_sensi <- df_diff |>
    group_by(module) |>
    do({
      sensi_trimmed_values <- trim_extreme_values(
        .$kurt_diff,
        trim_percent = sensi_trim_pct
      )
      data.frame(
        module = .$module[1],
        sensi_diff_trimmed = sensi_trimmed_values
      )
    })

  # Create QQ plots for kurt diff
  plt_qq_kurt <- ggplot(
    df_plot_trimmed_kurt, aes(sample = kurt_diff_trimmed)
  ) +
    stat_qq() +
    stat_qq_line() +
    facet_wrap(~module, scales = "free") +
    labs(
      title = paste0(model, " - QQ Plots of Kurtosis Differences by Module"),
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    ) +
    theme_minimal()
  ggsave(
    create.dir = TRUE,
    paste0("pdfs/qq_kurt_", model, ".pdf"),
    plot = plt_qq_kurt,
    width = 10,
    height = 6
  )

  # Create QQ plots for sensi diff
  plt_qq_sensi <- ggplot(
    df_plot_trimmed_sensi, aes(sample = sensi_diff_trimmed)
  ) +
    stat_qq() +
    stat_qq_line() +
    facet_wrap(~module, scales = "free") +
    labs(
      title = paste0(model, " - QQ Plots of Sensitivity Differences by Module"),
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    ) +
    theme_minimal()
  ggsave(
    paste0("pdfs/qq_sensi_", model, ".pdf"),
    plot = plt_qq_sensi,
    width = 10,
    height = 6
  )

  hist_fill_color <- "#66c2a5"
  density_line_color <- "#fc8d62"
  norm_line_color <- "blue"
  # Create histograms with normal curve overlay for trimmed data
  plt_hist_kurt <- ggplot(df_plot_trimmed_kurt, aes(x = kurt_diff_trimmed)) +
    geom_histogram(
      aes(y = after_stat(density)),
      bins = 10,
      fill = hist_fill_color,
      color = "black"
    ) +
    geom_density(color = density_line_color, linewidth = 1) +
    stat_function(
      fun = dnorm,
      args = list(
        mean = mean(df_plot_trimmed_kurt$kurt_diff_trimmed),
        sd = sd(df_plot_trimmed_kurt$kurt_diff_trimmed)
      ),
      color = norm_line_color, linewidth = 1, linetype = "dashed"
    ) +
    facet_wrap(~module, scales = "free") +
    labs(
      # title = paste0(
      #   model,
      #   " - Histograms of Kurtosis Differences with Normal Curve Overlay"
      # ),
      x = paste0(
        "Kurtosis Difference (",
        formatC(kurt_trim_pct * 100, format = "f", digits = 0),
        "% Trimmed)"
      ),
      y = "Density"
    ) +
    theme_minimal()
  ggsave(
    paste0("pdfs/hist-kurt-", model, ".pdf"),
    plot = plt_hist_kurt,
    width = 10,
    height = 6
  )

  # Create histograms with normal curve overlay for trimmed data
  plt_hist_sensi <- ggplot(df_plot_trimmed_sensi, aes(x = sensi_diff_trimmed)) +
    geom_histogram(
      aes(y = after_stat(density)),
      bins = 10,
      fill = hist_fill_color,
      color = "black"
    ) +
    geom_density(color = density_line_color, linewidth = 1) +
    stat_function(
      fun = dnorm,
      args = list(
        mean = mean(df_plot_trimmed_sensi$sensi_diff_trimmed),
        sd = sd(df_plot_trimmed_sensi$sensi_diff_trimmed)
      ),
      color = norm_line_color, linewidth = 1, linetype = "dashed"
    ) +
    facet_wrap(~module, scales = "free") +
    labs(
      # title = paste0(
      #   model,
      #   " - Histograms of Sensitivity Differences with Normal Curve Overlay"
      # ),
      x = paste0(
        "Sensitivity Difference (",
        formatC(sensi_trim_pct * 100, format = "f", digits = 0),
        "% Trimmed)"
      ),
      y = "Density"
    ) +
    theme_minimal()
  ggsave(
    paste0("pdfs/hist-sensi-", model, ".pdf"),
    plot = plt_hist_sensi,
    width = 10,
    height = 6
  )
}
