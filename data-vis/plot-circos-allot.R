#!/usr/bin/env Rscript

library(tidyverse)
library(readr)
library(optparse)
library(grid)
library(ComplexHeatmap)
library(circlize)
library(this.path)

# make reference to library function portable
source(file.path(here("functions"), "utils.R"))

plot_bar <- function(df_ppl_disp) {
  row_names <- df_ppl_disp$attempt
  df_ppl_disp <- df_ppl_disp |> select(!c("attempt"))
  rownames(df_ppl_disp) <- row_names
  my_colors <- c("lightblue", "mistyrose", "lightcyan")
  m_disp <- t(as.matrix(df_ppl_disp))
  the_bar_pos <- barplot(
    m_disp,
    col = my_colors,
    ylim = c(0, max(m_disp) + 1.5),
    axes = FALSE,
    width = 0.5,
    space = c(0.1, 0.5),
    beside = TRUE
  )

  # Add the legend outside the plot
  legend(
    "bottom",
    legend = rownames(m_disp),
    fill = my_colors,
    bty = "n",
    horiz = TRUE,
    inset = c(0, -0.3), # Adjust position to move outside the plot
    xpd = TRUE # Allow drawing outside the plot region
  )

  text(
    x = the_bar_pos,
    y = m_disp,
    label = formatC(m_disp, format = "f", digits = 2),
    pos = 3,
    offset = 0.5
  )
}

plot_allot_track <- function(df, model_id, color_map, the_attempt, the_sector) {
  circos.track(
    df$module,
    ylim = c(0, 1.05),
    bg.border = NA,
    panel.fun = function(x, y) {
      df_mod <- df |>
        filter(module == CELL_META$sector.index & attempt == the_attempt)

      if (CELL_META$sector.index == the_sector) {
        circos.text(
          CELL_META$xcenter,
          CELL_META$cell.ylim[2],
          paste0(the_attempt, "(", toupper(abbreviate(the_attempt)), ")"),
          cex = 0.7,
          facing = "bending.inside",
          niceFacing = TRUE
        )
      }
      for (i in 1:nrow(df_mod)) {
        row <- df_mod[i, ]
        color <- color_map[[row$cfg]]
        circos.rect(
          row$layer - 1 + 0.02, 0.05,
          row$layer - 0.02, 1,
          border = "darkgray",
          col = color
        )
      }
    }
  )
}

plot_allot_one <- function(
    df, df_ppl, model_id, color_map, attempt1, attempt2, attempt3, attempt4) {
  circos.clear()
  circos.par("track.height" = 0.09)
  circos.par("gap.degree" = 2)
  circos.initialize(df$module, x = df$layer)

  circos.track(
    ylim = c(0, 1),
    track.height = 0.07,
    panel.fun = function(x, y) {
      circos.text(
        CELL_META$xcenter,
        CELL_META$cell.ylim[2] - 0.7,
        CELL_META$sector.index,
        facing = "bending.inside",
        niceFacing = TRUE
      )
      circos.axis(
        labels.cex = 0.7,
        major.at = c(0, 5, 10, 15, 20, 25, 30, 35, 40)
      )
    }
  )
  plot_allot_track(df, model_id, color_map, attempt1, "self_attn.o_proj")
  plot_allot_track(df, model_id, color_map, attempt2, "self_attn.o_proj")
  plot_allot_track(df, model_id, color_map, attempt3, "self_attn.o_proj")
  plot_allot_track(df, model_id, color_map, attempt4, "self_attn.o_proj")

  text(0, 0, simplify_model_id(model_id), cex = 1.5, col = "darkblue")
  bpp <- unique(df_ppl$bpp)
  df_ppl_at1 <- df_ppl |> filter(attempt == attempt1)
  df_ppl_at2 <- df_ppl |> filter(attempt == attempt2)
  df_ppl_at3 <- df_ppl |> filter(attempt == attempt3)
  df_ppl_at4 <- df_ppl |> filter(attempt == attempt4)
  at1_ppl_wk <- formatC(df_ppl_at1$ppl_wikitext, format = "f", digits = 2)
  at1_ppl_mem <- formatC(df_ppl_at1$load_mem_allot, format = "f", digits = 2)
  at2_ppl_wk <- formatC(df_ppl_at2$ppl_wikitext, format = "f", digits = 2)
  at2_ppl_mem <- formatC(df_ppl_at2$load_mem_allot, format = "f", digits = 2)
  at3_ppl_wk <- formatC(df_ppl_at3$ppl_wikitext, format = "f", digits = 2)
  at3_ppl_mem <- formatC(df_ppl_at3$load_mem_allot, format = "f", digits = 2)
  at4_ppl_wk <- formatC(df_ppl_at4$ppl_wikitext, format = "f", digits = 2)
  at4_ppl_mem <- formatC(df_ppl_at4$load_mem_allot, format = "f", digits = 2)
  abbrv1 <- toupper(abbreviate(attempt1))
  abbrv2 <- toupper(abbreviate(attempt2))
  abbrv3 <- toupper(abbreviate(attempt3))
  abbrv4 <- toupper(abbreviate(attempt4))

  str_content <- paste0(
    "Bit Budget: ", bpp, "\n",
    abbrv1, ": PPL: ", at1_ppl_wk, " MEM: ", at1_ppl_mem, "\n",
    abbrv2, ": PPL: ", at2_ppl_wk, " MEM: ", at2_ppl_mem, "\n",
    abbrv3, ": PPL: ", at3_ppl_wk, " MEM: ", at3_ppl_mem, "\n",
    abbrv4, ": PPL: ", at4_ppl_wk, " MEM: ", at4_ppl_mem, "\n"
  )

  text(0, 0.25, str_content, cex = 0.8)
}

plot_allot_circos <- function(
    df_allot,
    df_ppl,
    model_id,
    budget,
    attempt1,
    attempt2,
    attempt3,
    attempt4) {
  model_cnt <- length(models)
  layout(matrix(1:model_cnt, nrow = 1, ncol = model_cnt))

  color_map <- list(
    b2g128 = "#2ca25f",
    b2g64 = "#41ae76",
    b2g32 = "#66c2a4",
    b3g128 = "#99d8c9",
    b3g64 = "#ccece6",
    b3g32 = "#e0ecf4",
    b4g128 = "#9ebcda",
    b4g64 = "#8c96c6",
    b4g32 = "#8c6bb1",
    b8g128 = "#88419d",
    b8g64 = "#810f7c",
    b8g32 = "#4d004b"
  )

  pdf(
    paste0("pdfs/circos-", model_id, "-", budget, ".pdf"),
    width = 10,
    height = 8
  )
  df_by_model <- df_allot |> filter(model == model_id)
  df_ppl_by_model <- df_ppl |> filter(model == model_id)
  plot_allot_one(
    df_by_model,
    df_ppl_by_model,
    model_id,
    color_map,
    attempt1,
    attempt2,
    attempt3,
    attempt4
  )
  circos.clear()

  lgd_grids <- Legend(
    at = names(color_map),
    type = "grid",
    ncol = 1,
    nrow = 12,
    legend_gp = gpar(fill = unlist(color_map, use.names = FALSE)),
    title_position = "topleft",
    title = "Config"
  )
  draw(
    packLegend(lgd_grids),
    x = unit(0.96, "npc"),
    y = unit(0.5, "npc"),
    just = c("right", "center")
  )
  circos.clear()
  dev.off()
}

strip_name <- function(name) {
  start <- nchar("fnorm-") + 1
  stop <- nchar(name) - 4
  return(substr(name, start, stop))
}

parser <- OptionParser()
parser <- add_option(
  parser, c("-m", "--model"),
  type = "character",
  help = "Model ID",
  metavar = "character"
)
parser <- add_option(
  parser, c("-b", "--budget"),
  type = "double",
  help = "Bit Budget",
  metavar = "double"
)
parser <- add_option(
  parser, c("-d", "--fnorm_data_dir"),
  type = "character",
  help = "Data directory of fnorm meta data",
  metavar = "character"
)
parser <- add_option(
  parser, c("-q", "--quant_cfg_allot_file"),
  type = "character",
  help = "The combined quant config allocation csv file",
  metavar = "character"
)
parser <- add_option(
  parser, c("-p", "--ppl_csv_file"),
  type = "character",
  help = "The combined PPL csv file",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt1"),
  type = "character",
  help = "The first attempt to plot",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt2"),
  type = "character",
  help = "The second attempt to plot",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt3"),
  type = "character",
  help = "The third attempt to plot",
  metavar = "character"
)
parser <- add_option(
  parser, c("--attempt4"),
  type = "character",
  help = "The fourth attempt to plot",
  metavar = "character"
)

args <- parse_args(parser)

if (is.null(args$model)) {
  model_id <- "Llama-2-7b-hf"
} else {
  model_id <- args$model
}

if (is.null(args$budget)) {
  budget <- 4.25
} else {
  budget <- args$budget
}

if (is.null(args$fnorm_data_dir)) {
  fnorm_data_dir <- "../src/data"
} else {
  fnorm_data_dir <- args$fnorm_data_dir
}

if (is.null(args$quant_cfg_allot_file)) {
  quant_cfg_allot_file <- "data/quant-cfg-allocation.csv"
} else {
  quant_cfg_allot_file <- args$quant_cfg_allot_file
}

if (is.null(args$ppl_csv_file)) {
  ppl_csv_file <- "data/combined.csv"
} else {
  ppl_csv_file <- args$ppl_csv_file
}

if (is.null(args$attempt1)) {
  attempt1 <- "mxq1"
} else {
  attempt1 <- args$attempt1
}
attempt2 <- args$attempt2
attempt3 <- args$attempt3
attempt4 <- args$attempt4

if (is.null(args$ppl_csv_file)) {
  ppl_csv_file <- "data/combined.csv"
} else {
  ppl_csv_file <- args$ppl_csv_file
}

fnorm_dir <- normalizePath(fnorm_data_dir)
fnorm_fps <- dir(
  path = fnorm_dir,
  pattern = paste0("fnorm-", model_id, "\\.csv$"),
  full.names = TRUE
)
print(fnorm_dir)
print(fnorm_fps)
names(fnorm_fps) <- sapply((basename(fnorm_fps)), strip_name)
df_fnorm <- plyr::ldply(
  fnorm_fps,
  read.csv,
  stringsAsFactors = FALSE,
  .id = "model"
)

k_cols <- c(
  "model", "module", "layer", "cfg", "fnorm", "sensitivity", "memmb", "params"
)
df_fnorm <- df_fnorm |>
  dplyr::mutate(
    cfg = paste0("b", nbit1, "g", gsize1)
  ) |>
  select(all_of(k_cols))

df_ppl_all <- read_csv(ppl_csv_file) |>
  dplyr::mutate(
    attempt = R.utils::toCamelCase(attempt, split = "-", capitalize = TRUE)
  )

df_cfgs <- read_csv(quant_cfg_allot_file)

df_cfg_by_budget <- df_cfgs |>
  filter(
    bit_budget == budget
  ) |>
  dplyr::mutate(
    cfg = paste0("b", b1, "g", g1)
  ) |>
  select(-c("b1", "g1", "b2", "g2", "bit_budget", "memmb")) |>
  left_join(
    df_fnorm,
    join_by(model == model, module == module, layer == layer, cfg == cfg)
  ) |>
  dplyr::mutate(
    cfg = factor(
      cfg,
      levels = c(
        "b2g128", "b2g64", "b2g32",
        "b3g128", "b3g64", "b3g32",
        "b4g128", "b4g64", "b4g32",
        "b8g128", "b8g64", "b8g32"
      )
    ),
    attempt = R.utils::toCamelCase(attempt, split = "-", capitalize = TRUE)
  )

models <- c(model_id)
df_ppl_disp <- df_ppl_all |>
  filter(
    bpp == budget &
      (attempt == attempt1 |
        attempt == attempt2 |
        attempt == attempt3 |
        attempt == attempt4
      )
  ) |>
  select(
    c(
      "model",
      "bpp",
      "attempt",
      "ppl_wikitext",
      "ppl_c4",
      "load_mem_allot"
    )
  )

for (model_id in models) {
  plot_allot_circos(
    df_cfg_by_budget,
    df_ppl_disp,
    model_id,
    budget,
    attempt1,
    attempt2,
    attempt3,
    attempt4
  )
}

# plot_bar(df_ppl_disp)
