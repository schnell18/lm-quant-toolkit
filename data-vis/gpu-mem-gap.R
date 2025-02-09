# Load required libraries
library(ggplot2)
library(ggrepel)

# Define the years to be included
years <- 2017:2025

# Create a data frame for GPU data (memory in GiB)
gpu_data <- data.frame(
  Year = years,
  Category = "GPU",
  # Representative Nvidia GPUs:
  # 2017: Titan Xp, 2018: Tesla V100, 2019: Quadro RTX 8000, 
  # 2020 & 2021: Nvidia A100, 2022: Nvidia H100, then hypothetical future GPUs.
  Name = c("Titan Xp", "Tesla V100", "Quadro RTX 8000", "Nvidia A100",
           "Nvidia A100", "Nvidia H100", "Nvidia H100", "Nvidia H200", "Nvidia B200"),
  Value = c(12, 32, 48, 80, 80, 80, 96, 141, 192)  # Memory in GiB
)

# Create a data frame for Model data (parameters in billions)
model_data <- data.frame(
  Year = years,
  Category = "Model",
  # Representative large models:
  # 2017: Transformer, 2018: BERT-large, 2019: GPT-2, 2020: GPT-3,
  # 2021: Megatron-Turing NLG, 2022: PaLM, then hypothetical future models.
  Name = c("Transformer", "BERT-large", "GPT-2", "GPT-3",
           "OPT", "PaLM", "GPT-4", "GPT-4o", "Grok 3 Ultra"),
  Value = c(0.07, 0.34, 1.5, 175, 530, 540, 1000, 1800, 3000)  # Parameters in billions
)

# Combine the two data frames
data <- rbind(gpu_data, model_data)
data$Category <- factor(data$Category, levels = c("GPU", "Model"))

# Create a label column to annotate the points:
# GPUs will be labeled with "Name (XXGB)" and Models with "Name (XXB)"
data$Label <- ifelse(data$Category == "GPU",
                     paste0(data$Name, " (", data$Value, "GB)"),
                     paste0(data$Name, " (", data$Value, "B)"))

# Generate the plot
p <- ggplot(data, aes(x = Year, y = Value, color = Category)) +
  # Plot the points
  geom_point(size = 3) +
  # Connect the points for each category with a dashed line
  geom_line(aes(group = Category), linetype = "dashed") +
  # Add non-overlapping text labels
  geom_text_repel(aes(label = Label), size = 3, show.legend = FALSE) +
  # Use a log10 scale on the y-axis
  
  scale_y_log10(
    name = "Model Size (Billion Parameters) / GPU Memory (GiB)",
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  scale_x_continuous(
    breaks = seq(2017, 2025, 1)
  ) +
  # Manually specify colors (a palette reminiscent of Nature figures)
  scale_color_manual(values = c("GPU" = "#1b9e77", "Model" = "#d95f02")) +
  labs(x = "Year",
       y = "Size (GB for GPUs; Billions of Parameters for Models)"
       #title = "Growing Gap Between GPU Memory and Model Size (2017-2025)",
       #subtitle = "Large neural model parameter counts vs. Representative Nvidia GPU memory",
       #caption = "Note: The parameter count of Grok 3 Ultra is projection"
       ) +
  theme_bw() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12),
        axis.title.x = element_blank(),
        legend.position = "bottom",
        legend.title = element_blank()
        )

# Display the plot
print(p)

ggsave("pdfs/gpu-mem-lag.pdf", plot = p, width = 10, height = 6)
