#!/bin/bash

# check if R is installed
which R > /dev/null
if [[ $? -ne 0 ]]; then
    echo "R is not installed!"
    exit 1
fi

which Rscript > /dev/null
if [[ $? -ne 0 ]]; then
    echo "Rscript is not installed!"
    exit 1
fi

Rscript -e '
install.packages(
    c(
        "dplyr",
        "ggbreak",
        "ggplot2",
        "ggthemes",
        "jsonlite",
        "kableExtra",
        "knitr",
        "openxlsx",
        "optparse",
        "patchwork",
        "plotly",
        "plyr",
        "pracma",
        "RColorBrewer",
        "readr",
        "safetensors",
        "stringr",
        "this.path",
        "tidyverse"
    )
)

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("ComplexHeatmap")

install.packages(
    "ggmagnify",
    repos = c(
        "https://hughjonesd.r-universe.dev",
        "https://cloud.r-project.org"
    )
)
'
