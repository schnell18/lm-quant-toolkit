library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(openxlsx)

save_fnorm_by_model <- function(df) {
  models <- unique(df$model)
  for (mdl in models) {
    df_by_model <- df |>
      filter(model == mdl) |>
      select(-c("model")) |>
      write_csv(paste0("fnorm-", mdl, ".csv"))
  }
}

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

df_com1 <- df_fnorm |>
  filter(
    model == "Llama-2-7b-hf"
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

df_com2_1 <- df_fnorm |>
  filter(
    model == "Meta-Llama-3-8B" & part != "attn_out"
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

# The Llama3-8B has no attn_out metric, use attn_in instead
df_sensi_attn_in <- df_sensi |>
  filter(part == "attn_in")

df_com2_2 <- df_fnorm |>
  filter(
    model == "Meta-Llama-3-8B" & part == "attn_out"
  ) |>
  left_join(
    df_sensi_attn_in,
    by = join_by(
      model == model,
      layer == layer,
      nbit1 == nbits,
      gsize1 == group_size
    )
  ) |>
  select(-c("dataset", "part.x", "part.y"))

df_com <- bind_rows(df_com1, df_com2_1, df_com2_2)
save_fnorm_by_model(df_com)
