library(tidyverse)
library(ggthemes)
library(readr)

df_ppl_bits <- read_csv("../kube-sft/data/ppl-bits.csv")
metrics <- c("ppl_wikitext", "ppl_c4", "ppl_ptb")
all_cols <- c("model", "bits", metrics)
df_ppl_bits <- df_ppl_bits |>
  select(all_of(all_cols)) |>
  pivot_longer(
    cols = metrics,
    names_to = c(".value", "dataset"),
    names_sep = "_"
  )

df_disp <- df_ppl_bits |>
  filter(
    !grepl("ptb", dataset)
  ) |>
  filter(
    bits >= 3.0
  )
ggplot(df_disp, aes(x = bits, y = ppl)) +
  geom_point(aes(shape = model, color = model, y = ppl)) +
  geom_line(aes(color = model, y = ppl)) +
  # ylim(4, 17) +
  # geom_hline(yintercept = max_acc1 * 0.99, linetype = "dotted", color = "blue") +
  # geom_hline(yintercept = max_acc1, linetype = "dotted", color = "blue") +
  # annotate("text", x = 14, y = max_acc1 * 1.01, label = "0-shot Top-1 FP16") +
  labs(x = "Bit Budget", y = "Perplexity") +
  theme_gray(base_size = 16) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16)
  ) +
  facet_wrap(~dataset, scales = "free") +
  scale_color_solarized()
ggsave("pdfs/mxq-wikitext-ppls.pdf", width = 8, height = 5)
