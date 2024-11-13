library(tidyverse)
library(stringr)


calc_bpp <- function(config) {
  if (config == "base") {
    return(16.0)
  } else if (startsWith(config, "b")) {
    b1 <- strtoi(substr(config, 2, 2))
    g1 <- strtoi(substr(config, 4, nchar(config)))
    b2 <- 8
    g2 <- 128
    return(round(b1 + 2 * b2 / g1 + 32 / g1 / g2, digits = 2))
  } else {
    return(as.numeric(sub("_", ".", config)))
  }
}

zsc_csvs <- c(
  "data/vit/result-eval_zeroshot_cls_fp16-20240919221324.csv",
  "data/vit/result-eval_zs_B_hqq-20240924115024.csv",
  "data/vit/result-eval_zs_H_hqq-20240922075536.csv",
  "data/vit/result-eval_zs_BH_20_mxq-20240930160208.csv",
  "data/vit/result-eval_zs_BH_b2_4-mxq-20241001224316.csv"
)

lnp_csvs <- c(
  "data/vit/result-eval_lp_BLH_fp16-20240927145518.csv",
  "data/vit/result-eval_lp_H_hqq-20240922062756.csv",
  "data/vit/result-eval_lp_B_hqq-20240927013957.csv",
  "data/vit/result-eval_lp_BH_20_mxq-20240930032213.csv",
  "data/vit/result-eval_lp_BH_b2_4-mxq-20241001220858.csv"
)

zscs <- list()
for (zsc_csv in zsc_csvs) {
  zsc <- read_csv(
    zsc_csv,
    col_select = c(
      model, algo, config,
      zeroshot_mem_allot, zeroshot_mem_reserved,
      acc1_zeroshot_cls, acc5_zeroshot_cls,
      recall_zeroshot_cls, duration_zeroshot_cls
    )
  ) |>
    mutate(
      zeroshot_mem_allot = zeroshot_mem_allot / 1024 / 1024,
      zeroshot_mem_reserved = zeroshot_mem_reserved / 1024 / 1024,
      bpp = sapply(config, calc_bpp)
    ) |>
    relocate(bpp, .after = config)
  zscs <- append(zscs, list(zsc))
}
combined_zsc <- bind_rows(zscs)


lnps <- list()
for (lnp_csv in lnp_csvs) {
  lnp <- read_csv(lnp_csv) |>
    select(
      model, algo, config,
      linear_probe_mem_allot, linear_probe_mem_reserved,
      acc1_linear_probe, acc5_linear_probe,
      recall_linear_probe, duration_linear_probe
    ) |>
    mutate(
      linear_probe_mem_allot = linear_probe_mem_allot / 1024 / 1024,
      linear_probe_mem_reserved = linear_probe_mem_reserved / 1024 / 1024
    )
  lnps <- append(lnps, list(lnp))
}
combined_lnp <- bind_rows(lnps)

combined <- combined_zsc |>
  left_join(combined_lnp, join_by(model, algo, config)) |>
  arrange(model, algo, config)
write_csv(combined, "data/vit/combined.csv")
