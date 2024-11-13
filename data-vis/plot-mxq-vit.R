library(tidyverse)
library(ggthemes)
library(readr)


zs_all_cols <- c(
  "model", "algo", "config",
  "bpp", "zeroshot_mem_allot", "acc1_zeroshot_cls",
  "acc5_zeroshot_cls", "recall_zeroshot_cls", "duration_zeroshot_cls"
)

lp_all_cols <- c(
  "model", "algo", "config",
  "bpp", "linear_probe_mem_allot", "acc1_linear_probe",
  "acc5_linear_probe", "recall_linear_probe", "duration_linear_probe"
)

df_all <- read_csv("data/vit/combined.csv")
df_zeroshot <- df_all |>
  select(all_of(zs_all_cols)) |>
  rename(
    mem = zeroshot_mem_allot,
    acc1 = acc1_zeroshot_cls,
    acc5 = acc5_zeroshot_cls,
    recall = recall_zeroshot_cls,
    duration = duration_zeroshot_cls
  )

df_linear_probe <- df_all |>
  select(all_of(lp_all_cols)) |>
  rename(
    mem = linear_probe_mem_allot,
    acc1 = acc1_linear_probe,
    acc5 = acc5_linear_probe,
    recall = recall_linear_probe,
    duration = duration_linear_probe
  )

zeroshot_h <- df_zeroshot |>
  filter(
    grepl("ViT-H-14", model)
  )
max_acc1 <- max(zeroshot_h$acc1)
max_acc5 <- max(zeroshot_h$acc5)
ggplot(zeroshot_h, aes(x = bpp, y = acc1)) +
  geom_point(size = 2.5, aes(shape = algo, color = algo, y = acc1)) +
  ylim(max_acc1 * 0.95, max_acc5 * 1.01) +
  geom_hline(yintercept = max_acc1 * 0.99, linetype = "dotted", color = "blue") +
  geom_hline(yintercept = max_acc1, linetype = "dotted", color = "blue") +
  annotate("text", x = 14, y = max_acc1 * 1.01, label = "0-shot Top-1 FP16") +
  geom_point(size = 2.5, aes(shape = algo, color = algo, y = acc5)) +
  geom_hline(yintercept = max_acc5 * 0.99, linetype = "dotted", color = "blue") +
  geom_hline(yintercept = max_acc5, linetype = "dotted", color = "blue") +
  annotate("text", x = 14, y = max_acc5 * 1.01, label = "0-shot Top-5 FP16") +
  scale_x_continuous(trans = "log2") +
  labs(x = "Bit Budget", y = "Zero-shot Classification Accuracy") +
  theme_gray(base_size = 16) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16)
  ) +
  # facet_wrap(~model, scales = "free") +
  scale_color_solarized()
ggsave("pdfs/mxq-vit-h-14-zs.pdf", width = 8, height = 6)

linear_probe_h <- df_linear_probe |>
  filter(
    grepl("ViT-H-14", model)
  )
bmax_acc1 <- max(linear_probe_h$acc1)
bmax_acc5 <- max(linear_probe_h$acc5)
ggplot(linear_probe_h, aes(x = bpp, y = acc1)) +
  geom_point(size = 2.5, aes(shape = algo, color = algo, y = acc1)) +
  ylim(bmax_acc1 * 0.95, bmax_acc5 * 1.01) +
  geom_hline(yintercept = bmax_acc1 * 0.99, linetype = "dotted", color = "blue") +
  geom_hline(yintercept = bmax_acc1, linetype = "dotted", color = "blue") +
  annotate("text", x = 14, y = bmax_acc1 * 1.01, label = "Linear Top-1 FP16") +
  geom_point(size = 2.5, aes(shape = algo, color = algo, y = acc5)) +
  geom_hline(yintercept = bmax_acc5 * 0.99, linetype = "dotted", color = "blue") +
  geom_hline(yintercept = bmax_acc5, linetype = "dotted", color = "blue") +
  annotate("text", x = 14, y = bmax_acc5 * 1.01, label = "Linear Top-5 FP16") +
  scale_x_continuous(trans = "log2") +
  labs(x = "Bit Budget", y = "Linear Probe Accuracy") +
  theme_gray(base_size = 16) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16)
  ) +
  # facet_wrap(~model, scales = "free") +
  scale_color_solarized()
ggsave("pdfs/mxq-vit-h-14-lp.pdf", width = 8, height = 6)

# geom_point(aes(size = mem, color = algo)) +
# geom_smooth() +
