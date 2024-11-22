#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggbreak)
library(readr)
library(patchwork)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  csv_fp <- "data/combined.csv"
} else {
  csv_fp <- args[1]
}

all_cols <- c(
  "model", "algo", "config", "attempt",
  "bpp", "ppl_wikitext", "ppl_c4"
)
df_all <- read_csv(csv_fp) |>
  select(all_of(all_cols)) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Meta-Llama-3-8B", "Llama-2-13b-hf"),
      labels = c("Llama-2-7B", "Llama-3-8B", "Llama-2-13B")
    ),
    algo = factor(
      algo,
      levels = c("mxq", "fp16", "awq", "gptq", "bnb", "hqq"),
      labels = c("MXQ", "FP16", "AWQ", "GPTQ", "BnB", "HQQ"),
    ),
    attempt = factor(
      attempt,
      levels = c(
        "mxq1",
        "head-prioritized_1_05",
        "head-prioritized_1_10",
        "tail-prioritized_1_05",
        "tail-prioritized_1_10",
        "tail-prioritized_1_15",
        "tail-prioritized_1_20",
        "kurt-scaled"
      ),
      labels = c(
        "MXQ1",
        "HP105",
        "HP110",
        "TP105",
        "TP110",
        "TP115",
        "TP120",
        "KURT-SCALED"
      ),
    )
  )

df_wikitxt_all <- df_all |>
  rename(ppl = ppl_wikitext)
df_c4_all <- df_all |>
  rename(ppl = ppl_c4)


# Plot Llama-2-7b memory drop vs PPL loss ---------------------------------

model_name <- "Llama-2-7B"
df_wikitxt <- df_wikitxt_all |>
  filter(
    grepl(model_name, model) & bpp >= 2.5
  )
min_ppl <- min(df_wikitxt$ppl)
min_bpp <- min(df_wikitxt$bpp)
plt5 <- ggplot(
  subset(df_wikitxt, algo != "MXQ"),
  aes(x = bpp, y = ppl),
) +
  geom_point(
    data = subset(df_wikitxt, algo == "MXQ"),
    size = 0.5,
    aes(shape = algo, color = attempt, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memery Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(5.18, 5.18 * 1.20, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 5.18) / 5.18, name = "% Degradation")
  ) +
  labs(x = "WikiText2", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Attempt:")
  ) +
  theme(legend.position = "none") +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()

df_c4 <- df_c4_all |>
  filter(
    grepl(model_name, model) & bpp >= 2.5
  )
min_ppl <- min(df_c4$ppl)
min_bpp <- min(df_c4$bpp)
plt6 <- ggplot(
  subset(df_c4, algo != "MXQ"),
  aes(x = bpp, y = ppl),
) +
  geom_point(
    data = subset(df_c4, algo == "MXQ"),
    size = 0.5,
    aes(shape = algo, color = attempt, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16)
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(6.95, 6.95 * 1.20, 0.25),
    sec.axis = sec_axis(~ 100 * (. - 6.95) / 6.95, name = "% Degradation")
  ) +
  labs(x = "C4", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Attempt:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()

final_plot1 <- plt5 / plt6 + plot_layout(heights = c(2, 3))
pdf.options(reset = TRUE, onefile = FALSE)
ggsave(
  paste0("pdfs/", "mxq-", model_name, ".pdf"),
  plot = final_plot1, width = 10, height = 7
)

# Plot Llama-2-13b memory drop vs PPL loss ---------------------------------

model_name <- "Llama-2-13B"
df_wikitxt <- df_wikitxt_all |>
  filter(
    model == model_name & bpp >= 2.5
  )
min_ppl <- min(df_wikitxt$ppl)
min_bpp <- min(df_wikitxt$bpp)
plt1 <- ggplot(
  subset(df_wikitxt, algo != "MXQ"),
  aes(x = bpp, y = ppl),
) +
  geom_point(
    data = subset(df_wikitxt, algo == "MXQ"),
    size = 0.5,
    aes(shape = algo, color = attempt, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memery Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(4.63, 4.63 * 1.20, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 4.63) / 4.63, name = "% Degradation")
  ) +
  labs(x = "WikiText2", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Attempt:")
  ) +
  theme(
    legend.position = "none"
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()

df_c4 <- df_c4_all |>
  filter(
    grepl(model_name, model) & bpp >= 2.5
  )
min_ppl <- min(df_c4$ppl)
min_bpp <- min(df_c4$bpp)
plt2 <- ggplot(
  subset(df_c4, algo != "MXQ"),
  aes(x = bpp, y = ppl),
) +
  geom_point(
    data = subset(df_c4, algo == "MXQ"),
    size = 0.5,
    aes(shape = algo, color = attempt, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16)
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(6.45, 6.45 * 1.20, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 6.45) / 6.45, name = "% Degradation")
  ) +
  labs(x = "C4", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Attempt:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()

final_plot2 <- plt1 / plt2 + plot_layout(heights = c(2, 3))
pdf.options(reset = TRUE, onefile = FALSE)
ggsave(
  paste0("pdfs/", "mxq-", model_name, ".pdf"),
  plot = final_plot2, width = 10, height = 7
)


# Plot Llama-3-8B memory drop vs PPL loss ---------------------------------

model_name <- "Llama-3-8B"
df_wikitxt <- df_wikitxt_all |>
  filter(
    grepl(model_name, model) & bpp >= 2.5
  )
min_ppl <- round(min(df_wikitxt$ppl), digits = 2)
min_bpp <- round(min(df_wikitxt$bpp), digits = 2)
plt3 <- ggplot(
  subset(df_wikitxt, algo != "MXQ"),
  aes(x = bpp, y = ppl),
) +
  geom_point(
    data = subset(df_wikitxt, algo == "MXQ"),
    size = 0.5,
    aes(shape = algo, color = attempt, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memery Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.35),
    breaks = seq(min_ppl, min_ppl * 1.35, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 5.81) / 5.81, name = "% Degradation")
  ) +
  labs(x = "WikiText2", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Attempt:")
  ) +
  theme(
    legend.position = "none"
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()

df_c4 <- df_c4_all |>
  filter(
    grepl(model_name, model) & bpp >= 2.5
  )
min_ppl <- round(min(df_c4$ppl), digits = 2)
min_bpp <- round(min(df_c4$bpp), digits = 2)
plt4 <- ggplot(
  subset(df_c4, algo != "MXQ"),
  aes(x = bpp, y = ppl),
) +
  geom_point(
    data = subset(df_c4, algo == "MXQ"),
    size = 0.5,
    aes(shape = algo, color = attempt, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = "blue"
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16)
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.35),
    breaks = seq(min_ppl, min_ppl * 1.35, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 8.98) / 8.98, name = "% Degradation")
  ) +
  labs(x = "C4", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Attempt:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()

final_plot3 <- plt3 / plt4 + plot_layout(heights = c(4, 5))
pdf.options(reset = TRUE, onefile = FALSE)
ggsave(
  paste0("pdfs/", "mxq-", model_name, ".pdf"),
  plot = final_plot3, width = 12, height = 9
)
