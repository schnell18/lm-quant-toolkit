library(tidyverse)
library(plyr)
library(dplyr)
library(readr)

factor <- 5

df_sensi_27b <- read_csv("../src/data/fnorm-Llama-2-7b-hf.csv") |>
  mutate(
    sensitivity = ifelse(layer == 1, factor, 1.0)
  ) |>
  mutate(
    sensitivity = ifelse(layer == 31, factor, 1.0)
  )
write_csv(df_sensi_27b, "../src/data/fnorm-Llama-2-7b-hf.csv")

df_sensi_38b <- read_csv("../src/data/fnorm-Meta-Llama-3-8B.csv") |>
  mutate(
    sensitivity = ifelse(layer == 1, factor, 1.0)
  ) |>
  mutate(
    sensitivity = ifelse(layer == 31, factor, 1.0)
  )
write_csv(df_sensi_38b, "../src/data/fnorm-Meta-Llama-3-8B.csv")


df_sensi_213b <- read_csv("../src/data/fnorm-Llama-2-13b-hf.csv") |>
  mutate(
    sensitivity = ifelse(layer == 3, factor, 1.0)
  ) |>
  mutate(
    sensitivity = ifelse(layer == 39, factor, 1.0)
  )
write_csv(df_sensi_213b, "../src/data/fnorm-Llama-2-13b-hf.csv")
