# Load required library
library(ggplot2)

# Create a function for each distribution type
mesokurtic <- function(x) {
  dnorm(x, mean = 0, sd = 1)
}

leptokurtic <- function(x) {
  dnorm(x, mean = 0, sd = 0.8) * 1.3 # Reduced scaling factor
}

platykurtic <- function(x) {
  dnorm(x, mean = 0, sd = 1.6) * 0.85
}

# Create sequence of x values
x <- seq(-4, 4, length.out = 200)

# Create data frame for plotting
plot_data <- data.frame(
  x = rep(x, 3),
  y = c(
    leptokurtic(x),
    mesokurtic(x),
    platykurtic(x)
  ),
  Distribution = factor(
    rep(
      c(
        "Leptokurtic",
        "Mesokurtic",
        "Platykurtic"
      ),
      each = length(x)
    ),
    levels = c(
      "Leptokurtic",
      "Mesokurtic",
      "Platykurtic"
    )
  )
)

# Create the plot
plt <- ggplot() +
  # Add the distribution curves
  geom_line(
    data = plot_data,
    aes(x = x, y = y, color = Distribution),
    size = 1
  ) +
  
  # Customize colors
  scale_color_manual(values = c(
    "Leptokurtic" = "#007bff",
    "Mesokurtic" = "#28a745",
    "Platykurtic" = "#e83e8c"
  )) +
  # Customize labels and theme
  labs(
    x = "X",
    y = "Probability density",
    color = NULL
  ) +
  theme_minimal() +
  theme(
    axis.title.y = element_text(angle = 90),
    legend.position = "bottom",
    legend.text = element_text(size = 12),
  ) +
  # Set appropriate plot boundaries with higher y-limit
  coord_cartesian(xlim = c(-3.5, 3.5), ylim = c(0, 0.65))

ggsave(
  "pdfs/kurtosis.pdf",
  plot = plt,
  create.dir = TRUE,
  height = 4,
  width = 5
)
