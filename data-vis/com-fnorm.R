library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(openxlsx)

map_to_part <- function(module) {
  if (module == "self_attn.q_proj" || module == "self_attn.k_proj" ||
    module == "self_attn.v_proj") {
    return("attn_in")
  } else if (module == "self_attn.o_proj") {
    return("attn_out")
  } else if (module == "mlp.gate_proj" || module == "mlp.up_proj") {
    return("mlp_gate")
  } else if (module == "mlp.down_proj") {
    return("mlp_down")
  }
}

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
df_fnorm <- df_fnorm |>
  mutate(
    part = sapply(module, map_to_part)
  )

df_sensi <- read_csv("data/llama-sensitivity.csv") |>
  filter(
    dataset == "pileval"
  )

df_com <- df_fnorm |>
  filter(
    model == "Llama-2-7b-hf" | model == "Meta-Llama-3-8B"
  ) |>
  left_join(
    df_sensi,
    by = join_by(
      model == model,
      part == part,
      layer == layer,
      nbit1 == nbits,
      gsize1 == group_size
    )
  ) |>
  select(-c("dataset", "part"))
