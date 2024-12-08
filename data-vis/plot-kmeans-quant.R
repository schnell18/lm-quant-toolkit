# Load required libraries
library(ggplot2)
library(dplyr)
library(safetensors)
library(jsonlite)

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
base_dir <- "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
base_dir <- path.expand(base_dir)
wo <- get_tensor(orig_matrix, base_dir)
wo <- as.matrix(wo)


bs <- 64
cx <- 0
cy <- 0
ret <- get_region(cx, cy, bs)
wo1 <- wo[ret$sxs:ret$sxe, ret$sys:ret$sye]

# Generate data with mean = 0.25 (middle of [-1, 1.5]) and sd = 0.5
# Then clip to desired range
raw_data <- as.vector(wo1)
data <- data.frame(
  x = raw_data
)

# Perform k-means clustering
kmeans_result <- kmeans(data, centers = 16, nstart = 25)

# Add cluster assignments to the data
data$cluster <- as.factor(kmeans_result$cluster)

# Create a data frame for centroids
centroids <- data.frame(
  x = kmeans_result$centers[, 1],
  y = 0 # Set y to 0 for 1D visualization
)

# Create a jittered y-coordinate for better visualization
data$y <- jitter(rep(0, nrow(data)), amount = 0.3)

# Create the plot
p_kmeans <- ggplot() +
  # Plot the points with jittering
  geom_point(
    data = data,
    aes(x = x, y = y, color = cluster),
    alpha = 0.6,
    size = 2
  ) +
  # Add centroids
  geom_point(
    data = centroids,
    aes(x = x, y = y),
    color = "black",
    size = 3,
    shape = 2
  ) +
  # Add lines to show the actual 1D nature of data
  geom_segment(
    data = centroids,
    aes(x = x, xend = x, y = -0.5, yend = 0.5),
    color = "black",
    linetype = "dashed",
    alpha = 0.5
  ) +
  # Customize the theme and labels
  theme_minimal() +
  labs(
    title = "1D K-means Clustering (k=16)",
    subtitle = paste0("Llama2-7b ", matrix),
    x = "Value",
    y = ""
  ) +
  theme(
    legend.position = "none",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  scale_color_discrete(name = "Cluster")

p_kmeans
