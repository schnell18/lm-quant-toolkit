library(dplyr)
library(readr)

strip_name <- function(name) {
  start <- nchar("fnorm-") + 1
  stop <- nchar(name) - 4
  return(substr(name, start, stop))
}

fnorm_data_dir <- "../src/data/tmp"
fnorm_dir <- normalizePath(fnorm_data_dir)
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = paste0("fnorm-.*", "\\.csv$"),
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- plyr::ldply(
  fnorm_fps,
  read.csv,
  stringsAsFactors = FALSE,
  .id = "model"
)

k_cols <- c(
  "model",
  "module",
  "layer",
  "cfg",
  "memmb",
  "params",
  "fnorm",
  "sensitivity",
  "kurtosis"
)
df_fnorm <- df_fnorm |>
  dplyr::mutate(
    cfg = paste0("b", nbit1, "g", gsize1)
  ) |>
  select(all_of(k_cols))
write_csv(df_fnorm, "llama-fnorm.csv")
