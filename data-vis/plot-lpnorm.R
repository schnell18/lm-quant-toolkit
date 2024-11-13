# Load required libraries
library(plotly)

lp_norm_surface_fun <- function(p) {
  func <- function(x, y) {
    return((abs(x)^p + abs(y)^p)^(1 / p))
  }
  return(func)
}

# Generate a grid of points in 2D space
x <- seq(-10, 10, length.out = 100)
y <- seq(-10, 10, length.out = 100)

# Plotting the surface for p = 1/4
numerator <- 1
demonator <- 4
p1 <- plot_ly(
  x = x,
  y = y,
  z = outer(x, y, lp_norm_surface_fun(numerator / demonator)),
  type = "surface"
) |>
  layout(
    title = paste0("L_", numerator, "/", demonator, " Norm Surface"),
    scene = list(
      xaxis = list(title = "x"),
      yaxis = list(title = "y"),
      zaxis = list(title = "z", range = c(0, 200))
    )
  )

# Plotting the surface for p = 1/3
numerator <- 1
demonator <- 3
p2 <- plot_ly(
  x = x,
  y = y,
  z = outer(x, y, lp_norm_surface_fun(numerator / demonator)),
  type = "surface"
) |>
  layout(
    title = paste0("L_", numerator, "/", demonator, " Norm Surface"),
    scene = list(
      xaxis = list(title = "x"),
      yaxis = list(title = "y"),
      zaxis = list(title = "z", range = c(0, 95))
    )
  )

# Plotting the surface for p = 2
numerator <- 2
demonator <- 1
p3 <- plot_ly(
  x = x,
  y = y,
  z = outer(x, y, lp_norm_surface_fun(numerator / demonator)),
  type = "surface"
) |>
  layout(
    title = paste0("L_", numerator, " Norm Surface"),
    scene = list(
      xaxis = list(title = "x"),
      yaxis = list(title = "y"),
      zaxis = list(title = "z")
    )
  )

# Plotting the surface for p = 3
numerator <- 3
demonator <- 1
p4 <- plot_ly(
  x = x,
  y = y,
  z = outer(x, y, lp_norm_surface_fun(numerator / demonator)),
  type = "surface"
) |>
  layout(
    title = paste0("L_", numerator, " Norm Surface"),
    scene = list(
      xaxis = list(title = "x"),
      yaxis = list(title = "y"),
      zaxis = list(title = "z")
    )
  )

# Plotting the surface for p = 0.7 used by HQQ
numerator <- 7
demonator <- 10
p5 <- plot_ly(
  x = x,
  y = y,
  z = outer(x, y, lp_norm_surface_fun(numerator / demonator)),
  type = "surface"
) |>
  layout(
    # title = paste0("L_", numerator, "/", demonator, " Norm Surface"),
    scene = list(
      xaxis = list(title = "x"),
      yaxis = list(title = "y"),
      zaxis = list(title = "z", range = c(0, 35))
    )
  ) |>
  hide_colorbar() 

save_image(p5, "lpnorm-visual.pdf", weight = 1000, height = 600)
