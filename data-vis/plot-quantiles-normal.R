library(ggplot2)
library(ggthemes)
library(dplyr)

# Generate data for the normal distribution
x <- seq(-4, 4, length.out = 1000)
y <- dnorm(x)
df <- data.frame(x = x, y = y)

# Calculate quantiles
# Use ±20 for infinity
quantiles <- c(-20, qnorm(seq(0.05, 0.95, length.out = 15)), 20)

# Create data frame for the clipped vertical lines
line_data <- data.frame()
for (q in quantiles[2:(length(quantiles) - 1)]) { # Skip the ±20 points
  y_at_q <- dnorm(q)
  line_data <- rbind(
    line_data,
    data.frame(
      x = q,
      y_start = 0,
      y_end = y_at_q
    )
  )
}

# Create data for filled intervals
interval_data <- data.frame()
for (i in 1:(length(quantiles) - 1)) {
  x_seq <- seq(max(-4, quantiles[i]),
    min(4, quantiles[i + 1]),
    length.out = 100
  )
  interval_data <- rbind(
    interval_data,
    data.frame(
      x = x_seq,
      y = dnorm(x_seq),
      group = i
    )
  )
}

# Create the plot
p <- ggplot() +
  # Add filled intervals
  geom_ribbon(
    data = interval_data,
    aes(x = x, ymin = 0, ymax = y, group = group),
    fill = "lightgray",
    alpha = 0.3
  ) +
  # Add the normal distribution curve
  geom_line(
    data = df, aes(x = x, y = y),
    color = "black", size = 1
  ) +
  # Add thin dashed vertical lines for quantiles
  geom_segment(
    data = line_data,
    aes(
      x = x, xend = x,
      y = y_start, yend = y_end
    ),
    color = "black",
    linetype = "dashed",
    size = 0.3,
    alpha = 0.7
  ) +
  # Add quantile labels
  geom_text(
    data = data.frame(
      x = quantiles[2:(length(quantiles) - 1)], # Skip the ±20 points
      y = rep(-0.02, 15),
      label = paste0("", 1:15)
    ),
    aes(x = x, y = y, label = label),
    angle = -30,
    vjust = 1,
    size = 4
  ) +
  # Add infinity labels
  geom_text(
    data = data.frame(
      x = c(-4, 4),
      y = rep(-0.02, 2),
      label = c("0", "16")
    ),
    aes(x = x, y = y, label = label),
    angle = -30,
    vjust = 1,
    size = 4
  ) +
  # Customize the theme and labels
  theme_bw(base_size = 16) +
  theme(
    axis.line = element_line(color = "black"),
    plot.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank()
  ) +
  labs(
    x = "x",
    y = "Density"
  ) +
  # Set the axis limits
  scale_x_continuous(limits = c(-4, 4)) +
  ylim(-0.05, 0.45) +
  # Add theme customizations
  theme(
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )

# Save the plot as PDF
ggsave(
  "normal_quantiles.pdf",
  plot = p,
  width = 10, # Width in inches
  height = 7, # Height in inches
  device = "pdf",
  dpi = 300 # High resolution
)
