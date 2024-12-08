#!/usr/bin/env Rscript

library(tidyverse)
library(ggthemes)
library(ggbreak)
library(readr)

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
  filter(is.na(attempt) | attempt == "mxq1") |>
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
      levels = c("mxq1", "kurt-global", "kurt-scaled"),
      labels = c("MXQ1", "KURT-GLOBAL", "KURT-SCALED"),
    )
  )

df_wikitxt_all <- df_all |>
  rename(ppl = ppl_wikitext)
df_c4_all <- df_all |>
  rename(ppl = ppl_c4)

guideline_color <- "coral4"

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
    aes(shape = algo, color = algo, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, color = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memory Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(5.18, 5.18 * 1.20, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 5.18) / 5.18, name = "% Degradation")
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    color = guide_legend(title = "Method:")
  ) +
  theme(
    axis.title.x = element_blank(),
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()
plt5
pdf.options(reset = TRUE, onefile = FALSE)
ggsave(
  paste0("pdfs/", "mxq-wikitext-", model_name, ".pdf"),
  plot = plt5, width = 8, height = 5
)

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
    aes(shape = algo, color = algo, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memory Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(6.95, 6.95 * 1.20, 0.25),
    sec.axis = sec_axis(~ 100 * (. - 6.95) / 6.95, name = "% Degradation")
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    color = guide_legend(title = "Method:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()
plt6
ggsave(
  paste0("pdfs/", "mxq-c4-", model_name, ".pdf"),
  plot = plt6, width = 8, height = 6
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
    aes(shape = algo, color = algo, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memory Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(4.63, 4.63 * 1.20, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 4.63) / 4.63, name = "% Degradation")
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()
plt1
ggsave(
  paste0("pdfs/", "mxq-wikitext-", model_name, ".pdf"),
  plot = plt1, width = 8, height = 6
)

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
    aes(shape = algo, color = algo, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memory Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.20),
    breaks = seq(6.45, 6.45 * 1.20, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 6.45) / 6.45, name = "% Degradation")
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()
plt2
ggsave(
  paste0("pdfs/", "mxq-c4-", model_name, ".pdf"),
  plot = plt2, width = 8, height = 6
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
    aes(shape = algo, color = algo, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memory Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.35),
    breaks = seq(min_ppl, min_ppl * 1.35, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 5.81) / 5.81, name = "% Degradation")
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()
plt3
ggsave(
  paste0("pdfs/", "mxq-wikitext-", model_name, ".pdf"),
  plot = plt3, width = 8, height = 6
)

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
    aes(shape = algo, color = algo, y = ppl)
  ) +
  geom_point(size = 1.5, aes(shape = algo, y = ppl)) +
  geom_hline(
    yintercept = min_ppl * 1.02,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl * 1.01,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  geom_hline(
    yintercept = min_ppl,
    linetype = "dashed",
    size = 0.1,
    color = guideline_color
  ) +
  annotate("text", x = 15.8, y = min_ppl * 1.00, label = "FP16") +
  scale_x_break(c(5.5, 15.6)) +
  scale_x_continuous(
    limits = c(2.8, 16.2),
    breaks = seq(2.8, 5.5, 0.20),
    sec.axis = sec_axis(~ 100 * (16 - .) / 16, name = "% Memory Reduction")
  ) +
  scale_y_continuous(
    limits = c(min_ppl * 0.99, min_ppl * 1.35),
    breaks = seq(min_ppl, min_ppl * 1.35, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 8.98) / 8.98, name = "% Degradation")
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:")
  ) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14)
  ) +
  facet_wrap(~model, scales = "free") +
  scale_color_solarized()
plt4
ggsave(
  paste0("pdfs/", "mxq-c4-", model_name, ".pdf"),
  plot = plt4, width = 8, height = 6
)
