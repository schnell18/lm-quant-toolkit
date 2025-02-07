#!/usr/bin/env Rscript

library(tidyverse)
library(dplyr)
library(readr)

join_kurt <- function(model_id) {
  fnorm_csv_fp <- paste0("fnorm-", model_id, ".csv")
  kurt_csv_fp <- paste0("kurtosis-", model_id, ".csv")
  df1 <- read_csv(fnorm_csv_fp)
  df2 <- read_csv(kurt_csv_fp)
  df <- df1 |>
    left_join(df2, by = c("module", "layer"))
  write_csv(df, fnorm_csv_fp)
}

join_kurt("CLIP-ViT-B-32-laion2B-s34B-b79K")
join_kurt("CLIP-ViT-L-14-laion2B-s32B-b82K")
join_kurt("CLIP-ViT-H-14-laion2B-s32B-b79K")
