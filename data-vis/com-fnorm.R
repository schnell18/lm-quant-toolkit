library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(openxlsx)

strip_name <- function(name) {
  start <- nchar("fnorm-") + 1
  stop <- nchar(name) - 4
  return(substr(name, start, stop))
}

fnorm_dir <- path.expand("data/fnorm")
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = "fnorm-.*\\.csv$",
  full.names = TRUE
)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- ldply(fnorm_fps, read.csv, stringsAsFactors = FALSE, .id = "model")

write.xlsx(df_fnorm, "df_fnorm.xlsx", overwrite = TRUE, asTable = TRUE)
