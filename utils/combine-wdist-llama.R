library(dplyr)
library(readr)

strip_name <- function(name) {
  start <- nchar("wdist-") + 1
  stop <- nchar(name) - 4
  return(substr(name,
 start,
 stop))
}

wdist_data_dir <- "../data-vis/data/wdist/tmp"
wdist_dir <- normalizePath(wdist_data_dir)
wdist_fps <- dir(
  path = wdist_dir,

  pattern = paste0("wdist-.*",
 "\\.csv$"),

  full.names = TRUE
)
names(wdist_fps) <- sapply((basename(wdist_fps)),
 strip_name)
df_wdist <- plyr::ldply(
  wdist_fps,
  read.csv,
  stringsAsFactors = FALSE,
  .id = "model"
)

k_cols <- c(
  "model",
  "module",
  "param_count",
  "layer",
  "percentile_0",
  "percentile_99",
  "percentile_999",
  "percentile_9999",
  "percentile_100",
  "kurtosis"
)
df_wdist <- df_wdist |>
  select(all_of(k_cols))
write_csv(df_wdist, "llama-wdist.csv")
