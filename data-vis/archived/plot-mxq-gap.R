library(tidyverse)
library(ggthemes)
library(ggbreak)
library(readr)

all_cols <- c(
  "model", "algo", "config",
  "bpp", "ppl_wikitext", "ppl_c4"
)
df_all <- read_csv("data/combined.csv") |>
  select(all_of(all_cols)) |>
  filter(
    algo == "mxq" | algo == "pct5" | algo == "pct6" | algo == "fp16" | algo == "awq" | algo == "hqq"
  ) |>
  mutate(
    model = factor(
      model,
      levels = c("Llama-2-7b-hf", "Meta-Llama-3-8B", "Llama-2-13b-hf"),
      labels = c("Llama-2-7B", "Llama-3-8B", "Llama-2-13B")
    ),
    algo = factor(
      algo,
      levels = c("mxq", "pct5", "pct6", "fp16", "awq", "gptq", "bnb", "hqq"),
      labels = c("MXQ", "PCT5", "PCT6", "FP16", "AWQ", "GPTQ", "BnB", "HQQ"),
    )
  )

df_wikitxt_all <- df_all |>
  rename(ppl = ppl_wikitext)
df_c4_all <- df_all |>
  rename(ppl = ppl_c4)


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
  geom_point(size = 1.5, aes(shape = algo, color = algo, y = ppl)) +
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
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Method:")
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
  paste0("pdfs/", "ppl-wikitext-", model_name, ".pdf"),
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
  geom_point(size = 1.5, aes(shape = algo, color = algo, y = ppl)) +
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
    breaks = seq(6.45, 6.45 * 1.20, 0.20),
    sec.axis = sec_axis(~ 100 * (. - 6.45) / 6.45, name = "% Degradation")
  ) +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 14) +
  guides(
    shape = guide_legend(title = "Method:"),
    color = guide_legend(title = "Method:")
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
