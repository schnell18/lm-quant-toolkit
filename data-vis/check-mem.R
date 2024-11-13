library(ggplot2)
library(ggthemes)
library(openxlsx)
library(patchwork)
library(readr)
library(tidyverse)


df_mem <- read_csv("mxq-mem-bound-check.csv") |>
  mutate(
    memmb_delta = param_cnt * ((b1 + 2 * b2 / g1 + 32 / g1 / g2) - bit_budget) / 8 / 1024^2
  ) |>
  relocate(memmb_delta, .before = memmb) |>
  summarise(
    delta_tot = sum(memmb_delta),
    .by = c("model", "bit_budget", "attempt")
  )

df_mem2 <- read_csv("mxq-mem-bound-check.csv") |>
  summarise(
    memmb_tot = sum(memmb),
    param_quant_tot = sum(param_cnt),
    .by = c("model", "bit_budget", "attempt")
  ) |>
  mutate(
    theory_mem = param_quant_tot * bit_budget / 8 / 1024^2,
    within_bound = memmb_tot <= theory_mem
  )
