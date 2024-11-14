library(tidyverse)
library(ggthemes)
library(readr)

model_id <- "Llama-2-13b-hf"
wdist <- read_csv(paste0("data/wdist/wdist-", model_id, ".csv"))
percentiles <- c("0", "99", "99.9", "99.99", "100")
module_param_count <- wdist |>
  select(
    module, param_count
  ) |>
  group_by(module) |>
  summarise(
    param_count = sum(param_count)
  ) |>
  mutate(
    mod_disp = paste0(module, "(", formatC(param_count, big.mark = ","), ")")
  )

all_cols <- c("module", "layer", percentiles)
wdist <- wdist |>
  mutate(
    `0` = percentile_0,
    `99` = percentile_99 - percentile_0,
    `99.9` = percentile_999 - percentile_99,
    `99.99` = percentile_9999 - percentile_999,
    `100` = percentile_100 - percentile_9999,
  ) |>
  select(all_of(all_cols)) |>
  pivot_longer(
    cols = percentiles,
    names_to = "nth_percentile",
    names_transform = list(nth_percentile = as.numeric),
    values_to = "abs_val"
  ) |>
  mutate(
    nth_percentile = factor(nth_percentile, levels = rev(percentiles))
  ) |>
  left_join(module_param_count, by = c("module"))

ggplot(wdist, aes(x = layer, y = abs_val, fill = nth_percentile)) +
  geom_bar(stat = "identity", color = "gray50") +
  labs(x = "Layer", y = "Absolute Value", fill = "nth percentile") +
  # theme_gray(base_size = 16) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 12)
  ) +
  facet_wrap(~mod_disp, scales = "free") +
  scale_color_solarized()
ggsave(paste0("pdfs/", model_id, ".pdf"), width = 8, height = 6)
