#!/usr/bin/env Rscript

library(plotly)
library(safetensors)
library(jsonlite)
library(stringr)

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
}

# Helper function to determine the default cache directory
# based on environment variables
get_default_cache_dir <- function() {
  cache_dir <- Sys.getenv("HUGGINGFACE_HUB_CACHE")
  if (cache_dir != "") {
    return(cache_dir)
  }
  cache_dir <- Sys.getenv("TRANSFORMERS_CACHE")
  if (cache_dir != "") {
    return(cache_dir)
  }
  hf_home <- Sys.getenv("HF_HOME")
  if (hf_home != "") {
    return(hf_home)
  }
  xdg_cache_home <- Sys.getenv("XDG_CACHE_HOME")
  if (xdg_cache_home != "") {
    return(file.path(xdg_cache_home, "huggingface"))
  }
  return(file.path(Sys.getenv("HOME"), ".cache", "huggingface", "hub"))
}

# Main function to get the Hugging Face model storage base directory
get_hf_model_storge_base_dir <- function(model_id, hf_hub_dir = NULL) {
  # Replace slashes in model_id with double dashes
  model_id_x <- gsub("/", "--", model_id)

  # Determine cache directory: use hf_hub_dir if provided, otherwise get default
  if (!is.null(hf_hub_dir)) {
    cache_dir <- hf_hub_dir
  } else {
    cache_dir <- get_default_cache_dir()
  }

  # Construct the model directory path
  hf_model_dir <- file.path(cache_dir, paste0("models--", model_id_x))

  # Construct the path to the 'refs/main' file
  ref_main_fp <- file.path(hf_model_dir, "refs", "main")

  # Read the commit SHA from the 'refs/main' file and trim whitespace
  commit_sha <- readLines(ref_main_fp, n = 1, warn = FALSE)
  commit_sha <- trimws(commit_sha)

  # Construct and return the storage base directory path
  storage_base_dir <- file.path(hf_model_dir, "snapshots", commit_sha)
  return(storage_base_dir)
}

get_tensor <- function(
    layer,
    module,
    base_dir,
    index_json = "model.safetensors.index.json") {
  index_file <- file.path(base_dir, index_json)
  model_index <- fromJSON(index_file)

  matrix_name <- paste0("model.layers.", layer, ".", module, ".weight")
  if (exists(matrix_name, model_index$weight_map)) {
    st_file <- model_index$weight_map[[matrix_name]]
    st_file_fp <- file.path(base_dir, st_file)
    tensors <- safe_load_file(st_file_fp)
    return(tensors[[matrix_name]])
  }
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

# plot sub region of matrix -----------------------------------------------
plot_region <- function(
    model_id,
    layer,
    module,
    bs = 512,
    cx = 2533,
    cy = 3037) {
  title <- paste0(layer, ".", module)
  base_dir <- get_hf_model_storge_base_dir(model_id)
  wo <- get_tensor(layer, module, base_dir)
  wo <- as.matrix(wo)
  ret <- get_region(cx, cy, bs)
  wo1 <- wo[ret$sxs:ret$sxe, ret$sys:ret$sye]
  p1 <- create_3d_plot2(wo1, title)
  fig1 <- subplot(p1) |>
    layout(
      scene = list(
        domain = list(row = 1, column = 1),
        zaxis = list(title = "Weight", range = c(-1.7, 1.7)),
        camera = list(eye = list(x = 1.5, y = 1.2, z = 0.3)),
        aspectratio = list(x = 0.9, y = 0.85, z = 0.9)
      )
    )

  short_id <- str_split(model_id, "/", n = 2)[[1]][2]

  save_image(
    fig1,
    paste0("pdfs/", short_id, "-outlier-", layer, "_", module, ".pdf"),
    scale = 2
  )
  return(fig1)
}

layer <- 31
module <- "self_attn.o_proj"
model_id <- "meta-llama/Llama-2-7b-hf"
fig1 <- plot_region(model_id, layer, module)
layer <- 1
fig2 <- plot_region(model_id, layer, module)
