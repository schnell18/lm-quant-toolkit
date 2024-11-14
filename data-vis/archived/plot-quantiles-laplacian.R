library(ggplot2)
library(dplyr)

# Create Laplace distribution function
dlaplace <- function(x, mu = 0, b = 1) {
  1/(2*b) * exp(-abs(x - mu)/b)
}

# Generate data points
x <- seq(-4, 4, length.out = 1000)
y <- dlaplace(x)
df <- data.frame(x = x, y = y)

# Calculate quantiles using the Laplace quantile function
plaplace <- function(p, mu = 0, b = 1) {
  mu - b * sign(p - 0.5) * log(1 - 2 * abs(p - 0.5))
}

# Calculate quantiles
probs <- seq(0.05, 0.95, length.out = 15)
quantiles <- c(-20, sapply(probs, plaplace), 20)

# Create data frame for the clipped vertical lines
line_data <- data.frame()
for(q in quantiles[2:(length(quantiles)-1)]) {  # Skip the ±20 points
  y_at_q <- dlaplace(q)
  line_data <- rbind(line_data, 
                     data.frame(x = q, 
                                y_start = 0, 
                                y_end = y_at_q))
}

# Create data for filled intervals
interval_data <- data.frame()
for(i in 1:(length(quantiles)-1)) {
  x_seq <- seq(max(-4, quantiles[i]), 
               min(4, quantiles[i+1]), 
               length.out = 100)
  interval_data <- rbind(interval_data,
                         data.frame(
                           x = x_seq,
                           y = dlaplace(x_seq),
                           group = i
                         ))
}

# Create the plot
ggplot() +
  # Add filled intervals
  geom_ribbon(data = interval_data,
              aes(x = x, ymin = 0, ymax = y, group = group),
              fill = "lightblue",
              alpha = 0.3) +
  # Add the distribution curve
  geom_line(data = df, aes(x = x, y = y),
            color = "blue", size = 1) +
  # Add thin solid blue vertical lines for quantiles
  geom_segment(data = line_data,
               aes(x = x, xend = x, 
                   y = y_start, yend = y_end),
               color = "blue",
               linetype = "solid",
               size = 0.3,
               alpha = 0.7) +
  # Add quantile labels
  geom_text(data = data.frame(
    x = quantiles[2:(length(quantiles)-1)],  # Skip the ±20 points
    y = rep(-0.02, 15),
    label = paste0("Q", 1:15)
  ),
  aes(x = x, y = y, label = label),
  angle = 45,
  vjust = 1,
  size = 3) +
  # Add infinity labels
  geom_text(data = data.frame(
    x = c(-4, 4),
    y = rep(-0.02, 2),
    label = c("Q0 (-∞)", "Q16 (+∞)")
  ),
  aes(x = x, y = y, label = label),
  angle = 45,
  vjust = 1,
  size = 3) +
  # Customize the theme and labels
  theme_minimal() +
  labs(
    title = "Laplace Distribution with 17 Quantiles",
    subtitle = "μ = 0, b = 1",
    x = "x",
    y = "Density"
  ) +
  # Set the axis limits
  scale_x_continuous(limits = c(-4, 4)) +
  ylim(-0.05, 0.55) +
  # Add theme customizations
  theme(
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )
