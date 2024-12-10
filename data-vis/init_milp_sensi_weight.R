#!/usr/bin/env Rscript

library(tidyverse)
library(dplyr)
library(readr)

factor <- 5.0

sensi_factor_mapper <- function(layer1, layer2, factor) {
  map_to_sensi_factor <- function(layer) {
    return(ifelse(layer == layer1 || layer == layer2, factor, 1.0))
  }
  return(map_to_sensi_factor)
}

df_sensi_27b <- read_csv("../src/data/fnorm-Llama-2-7b-hf.csv") |>
  mutate(
    sensitivity = sapply(layer, sensi_factor_mapper(1, 31, factor))
  )
write_csv(df_sensi_27b, "../src/data/fnorm-Llama-2-7b-hf.csv")

df_sensi_38b <- read_csv("../src/data/fnorm-Meta-Llama-3-8B.csv") |>
  mutate(
    sensitivity = sapply(layer, sensi_factor_mapper(1, 31, factor))
  )
write_csv(df_sensi_38b, "../src/data/fnorm-Meta-Llama-3-8B.csv")


df_sensi_213b <- read_csv("../src/data/fnorm-Llama-2-13b-hf.csv") |>
  mutate(
    sensitivity = sapply(layer, sensi_factor_mapper(3, 39, factor))
  )
write_csv(df_sensi_213b, "../src/data/fnorm-Llama-2-13b-hf.csv")
