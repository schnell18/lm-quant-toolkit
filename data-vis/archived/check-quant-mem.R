library(ggplot2)
library(ggthemes)
library(openxlsx)
library(patchwork)
library(readr)
library(tidyverse)


calc_bpp <- function(b1, g1, b2, g2) {
  return(round(b1 + 2 * b2 / g1 + 32 / g1 / g2, digits = 2))
}

# df_cfg1 <- read_csv("data/mxq-quant-cfgs-mxq1.csv")
# df_cfg2 <- read_csv("data/kurt/global/llama-mxq-cfgs.csv")
# df_cfg3 <- read_csv("data/kurt/scaled/llama-mxq-cfgs.csv")
# df_cfg4 <- read_csv("data/mxq-quant-cfgs-mxq1-5pct-tol.csv")
# df_cfg5 <- read_csv("data/mxq-quant-cfgs-kurt-scaled-6pct-tol.csv")
# df_cfg1$attempt <- "MXQ1"
# df_cfg2$attempt <- "kurt-global"
# df_cfg3$attempt <- "kurt-scaled"
# df_cfg4$attempt <- "PCT5"
# df_cfg5$attempt <- "kscaled-pct6"
# df_cfg <- bind_rows(df_cfg1, df_cfg2, df_cfg3, df_cfg4, df_cfg5)

df_cfg <- read_csv("mxq-mem-bound-check.csv")

df_mem_sum <- df_cfg |>
  group_by(
    model, bit_budget, attempt
  ) |>
  summarise(
    mem_tot = sum(memmb)
  ) |>
  pivot_wider(
    names_from = "attempt",
    values_from = "mem_tot"
  ) |>
  ungroup()

df_7b <- read_csv("data/fnorm/fnorm-Llama-2-7b-hf.csv")
df_13b <- read_csv("data/fnorm/fnorm-Llama-2-13b-hf.csv")
df_8b <- read_csv("data/fnorm/fnorm-Meta-Llama-3-8B.csv")
df_7b$model <- "Llama-2-7b-hf"
df_13b$model <- "Llama-2-13b-hf"
df_8b$model <- "Meta-Llama-3-8B"
df_llama <- bind_rows(df_7b, df_13b, df_8b) |>
  mutate(
    bit_budget = calc_bpp(nbit1, gsize1, nbit2, gsize2)
  ) |>
  group_by(model, bit_budget) |>
  summarise(
    mem_tot = sum(memmb),
    param_tot = sum(params)
  )

df_mem <- df_mem_sum |>
  left_join(
    df_llama,
    by = c("model", "bit_budget")
  ) |>
  rename(
    hqq = mem_tot
  ) |>
  mutate(
    theory = param_tot * bit_budget / 8 / 1024^2,
  ) |>
  select(!c("param_tot"))

write.xlsx(df_mem, "df_mem.xlsx", overwrite = TRUE, asTable = TRUE)

mem_gap_grid <- function(df_mem, mod, show_legend = FALSE, show_x_label = FALSE) {
  df_disp <- df_mem |>
    filter(model == mod) |>
    pivot_longer(
      cols = c("mxq1", "pct5", "pct6", "kurt-global", "kurt-scaled", "hqq", "theory"),
      names_to = "attempt",
      values_to = "memory"
    )
  df_gap <- df_mem |>
    filter(model == mod) |>
    mutate(
      gap_in_pct = 100 * (theory - mxq1) / mxq1
    ) |>
    select(c("bit_budget", "gap_in_pct"))
  # Gap percentage line plot (on top)
  gap_line_plot <- ggplot(df_gap, aes(x = bit_budget, y = gap_in_pct)) +
    geom_line(color = "blue") +
    theme_gray(base_size = 12) +
    labs(y = "% Min Mem Gap") +
    theme_minimal() +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_blank()
    )

  # line plot (on bottom)
  line_plot <- ggplot(
    df_disp,
    aes(x = bit_budget, y = memory)
  ) +
    geom_line(aes(color = attempt)) +
    geom_point(aes(shape = attempt, color = attempt)) +
    labs(x = "Bit Budget", y = "Memory") +
    theme_gray(base_size = 12) +
    guides(color = guide_legend(ncol = 1)) +
    facet_wrap(~model, scales = "free", ncol = 1)
  if (!show_x_label) {
    line_plot <- line_plot +
      theme(
        axis.title.x = element_blank(),
        axis.text.x = element_blank()
      )
  }

  if (show_legend) {
    line_plot <- line_plot +
      theme(
        legend.position = "right",
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 12)
      ) +
      scale_color_solarized()
  } else {
    line_plot <- line_plot +
      theme(legend.position = "none") +
      scale_color_solarized()
  }
  # Combine the line and bar plot vertically
  combined_plot <- gap_line_plot / line_plot + plot_layout(heights = c(1, 3))
  return(combined_plot)
}

p1 <- mem_gap_grid(df_mem, "Llama-2-7b-hf")
p2 <- mem_gap_grid(df_mem, "Llama-2-13b-hf", show_x_label = TRUE)
p3 <- mem_gap_grid(df_mem, "Meta-Llama-3-8B", show_legend = TRUE)

final_plot1 <- (p1 | p2 | p3)
final_plot1
ggsave(
  paste0("pdfs/", "mxq-mem-gap.pdf"),
  plot = final_plot1, width = 16, height = 9
)
