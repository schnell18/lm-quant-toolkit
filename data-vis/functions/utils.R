calc_bpp <- function(config) {
  if (config == "base") {
    return(16.0)
  } else if (startsWith(config, "b")) {
    b1 <- strtoi(substr(config, 2, 2))
    g1 <- strtoi(substr(config, 4, nchar(config)))
    b2 <- 8
    g2 <- 128
    return(round(b1 + 2 * b2 / g1 + 32 / g1 / g2, digits = 6))
  } else {
    return(round(as.numeric(sub("_", ".", config)), digits = 6))
  }
}

# model = factor(
#   model,
#   levels = c("Llama-2-7b-hf", "Meta-Llama-3-8B", "Llama-2-13b-hf"),
#   labels = c("Llama-2-7B", "Llama-3-8B", "Llama-2-13B")
# )

simplify_model_id <- function(model_id) {
  if (model_id == "Llama-2-7b-hf") {
    return("Llama-2-7B")
  } else if (model_id == "Meta-Llama-3-8B") {
    return("Llama-3-8B")
  } else if (model_id == "Llama-2-13b-hf") {
    return("Llama-2-13B")
  }
}

budget_to_cfg <- function(budget) {
  if (budget == 3.13) {
    return("b3g128")
  } else if (budget == 3.25) {
    return("b3g64")
  } else if (budget == 3.51) {
    return("b3g32")
  } else if (budget == 4.13) {
    return("b4g128")
  } else if (budget == 4.25) {
    return("b4g64")
  } else if (budget == 4.51) {
    return("b4g32")
  } else if (budget == 8.13) {
    return("b8g128")
  } else if (budget == 8.25) {
    return("b8g64")
  } else if (budget == 8.51) {
    return("b8g32")
  } else if (budget == 2.13) {
    return("b2g128")
  } else if (budget == 2.25) {
    return("b2g64")
  } else if (budget == 2.51) {
    return("b2g32")
  } else {
    return(budget)
  }
}

boost_budget <- function(budget, stop = 1) {
  budget_list <- list(
    2.13, 2.25, 2.51, 3.13, 3.25, 3.51, 4.13, 4.25, 4.51, 8.13, 8.25, 8.51
  )
  len_budget <- length(budget_list)
  idx <- which(budget_list == budget)
  if (length(idx) > 0) {
    while (idx + stop > len_budget) {
      stop <- stop - 1
    }
    return(budget_list[[idx + stop]])
  }
  return(budget)
}

abbrev_sensi_kurt <- function(attempt) {
  attempt <- gsub("sensi-boost", "SB", attempt)
  attempt <- gsub("sensi-abl", "SBAB", attempt)
  attempt <- gsub("kurt-boost", "KB", attempt)
  attempt <- gsub("kurt-abl", "KBAB", attempt)
  attempt <- gsub("sensi-milp-abl", "SMAB", attempt)
  attempt <- gsub("kurt-milp-abl", "KMAB", attempt)
  attempt <- gsub("sensi-milp", "SM", attempt)
  attempt <- gsub("kurt-milp", "KM", attempt)
  attempt <- gsub("-", "", attempt)
  attempt
}

zipcat <- function(vc1, vc2) {
  stopifnot(length(vc1) == length(vc2))
  result <- character(2 * length(vc1))
  for (i in 1:length(vc1)) {
    result[2 * i - 1] <- vc1[i]
    result[2 * i] <- vc2[i]
  }
  return(result)
}
