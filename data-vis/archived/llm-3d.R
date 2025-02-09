#!/usr/bin/env Rscript

library(plotly)
library(safetensors)
library(jsonlite)

create_3d_plot2 <- function(weight, title) {
  d <- dim(weight)
  z <- weight
  x <- 1:d[1]
  y <- 1:d[2]

  plot_ly(
    x = x, y = y, z = z,
    type = "surface", alpha = 0.6,
    colorscale = "coloraxis",
    showscale = FALSE
  )
  # colorscale = list(c(-1, 0, 1), c("blue", "white", "green"))
}

get_tensor <- function(
    matrix_name,
    base_dir,
    index_json = "model.safetensors.index.json") {
  index_file <- file.path(base_dir, index_json)
  model_index <- fromJSON(index_file)

  if (exists(matrix_name, model_index$weight_map)) {
    st_file <- model_index$weight_map[[matrix_name]]
    st_file_fp <- file.path(base_dir, st_file)
    tensors <- safe_load_file(st_file_fp)
    return(tensors[[matrix_name]])
  }
}

get_tensor2 <- function(matrix_name, base_dir, st_file) {
  st_file_fp <- file.path(base_dir, st_file)
  tensors <- safe_load_file(st_file_fp)
  return(tensors[[matrix_name]])
}

get_region <- function(cx, cy, bs, upper_x = 4096, upper_y = 4096) {
  sxs <- cx - bs / 2 + 1
  sxe <- cx + bs / 2
  sxs <- if (sxs < 1) 1 else sxs
  sxe <- if (sxe > upper_x) upper_x else sxe
  sys <- cy - bs / 2 + 1
  sye <- cy + bs / 2
  sys <- if (sys < 1) 1 else sys
  sye <- if (sye > upper_y) upper_y else sye
  return(list(sxs = sxs, sxe = sxe, sys = sys, sye = sye))
}

matrix <- "31.self_attn.o_proj"

orig_matrix <- paste0("model.layers.", matrix, ".weight")
quant_matrix <- paste0("model.layers.", matrix, ".qweight")
base_dir <- "~/work/llm-quant/snapshots"
base_dir <- path.expand(base_dir)
st_file <- "Llama-2-7b-hf-cmp.safetensors"
wo <- get_tensor2(orig_matrix, base_dir, st_file)
wo <- as.matrix(wo)
wq <- get_tensor2(quant_matrix, base_dir, st_file)
wq <- as.matrix(wq)

# plot sub region of matrix -----------------------------------------------
bs <- 512
cx <- 2533
cy <- 3037
ret <- get_region(cx, cy, bs)
wo1 <- wo[ret$sxs:ret$sxe, ret$sys:ret$sye]
wq1 <- wq[ret$sxs:ret$sxe, ret$sys:ret$sye]
p1 <- create_3d_plot2(wo1, matrix)
p2 <- create_3d_plot2(wq1, matrix)

fig1 <- subplot(p1) |>
  layout(
    # title = "Llama-2-7B Original Weight",
    scene = list(
      domain = list(row = 1, column = 1),
      zaxis = list(title = "Weight", range = c(-1.7, 1.7)),
      camera = list(eye = list(x = 1.5, y = 1.2, z = 0.3)),
      aspectratio = list(x = 0.9, y = 0.85, z = 0.9)
    )
  )
fig2 <- subplot(p2) |>
  layout(
    # title = "Llama-2-7B Quantized Weight",
    scene = list(
      domain = list(row = 1, column = 1),
      zaxis = list(title = "Weight", range = c(-1.7, 1.7)),
      camera = list(eye = list(x = 1.5, y = 1.5, z = 0.3))
    )
  )
save_image(fig1, "pdfs/llama2-7b-3D.pdf", weight = 1400, height = 1000)
save_image(fig2, "pdfs/llama2-7b-3D-quant.pdf", scale = 2)
save_image(fig1, "pdfs/llama2-7b-3D-orig.pdf", scale = 3)
