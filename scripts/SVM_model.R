# Build feature dataset: Spatial uLBP (2x2) + HOG + HSV hist
# Outputs:
# - outputs/features_spatial_ulbp_hog_hsv.csv
# - outputs/features_train.csv
# - outputs/features_test.csv
# - outputs/preprocess_excluded_files.csv
# - outputs/confusion_svm_linear.csv
# - outputs/confusion_svm_rbf.csv
# - outputs/metrics_svm_linear.csv
# - outputs/metrics_svm_rbf.csv
# - outputs/predictions_svm_linear.csv
# - outputs/predictions_svm_rbf.csv
# - outputs/svm_summary.txt

# Libraries
library(imager)
library(OpenImageR)
library(dplyr)
library(purrr)
library(readr)
library(tibble)
library(stringr)
library(caret)
library(digest)
library(nnet)
library(e1071)
library(here)

# Paths
data_dir   <- "data"
output_dir <- "outputs"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Parameters
IMG_SIZE <- 128        # image resize
H_BINS   <- 8          # HSV Hue bins
S_BINS   <- 4          # HSV Saturation bins
V_BINS   <- 4          # HSV Value bins
HOG_ORIENTATIONS <- 9  # HOG Orientations
HOG_CELLS <- 4         # HOG cells (4 => 4x4 cells for 128x128 in OpenImageR HOG)
TEST_P <- 0.30         # test split proportion
SEED <- 123            # reproducibility seed


# 1) DATA PRE-PROCESSING

# Collect image paths + labels
img_df <- tibble(
  path  = list.files(data_dir, recursive = TRUE, full.names = TRUE),
  label = basename(dirname(path))
) %>%
  mutate(label = tolower(label)) %>%
  filter(str_detect(tolower(path), "\\.(jpg|jpeg|png)$")) %>%
  filter(label %in% c("helmet", "no_helmet")) %>%
  distinct(path, .keep_all = TRUE)

stopifnot(nrow(img_df) > 0)
message("Initial images found: ", nrow(img_df))
print(table(img_df$label))

# Safe image load
safe_load_image <- purrr::possibly(
  function(p) imager::load.image(p),
  otherwise = NULL
)

# Basic file validity check
validity <- map_lgl(img_df$path, function(p) {
  img <- safe_load_image(p)
  !is.null(img) && prod(dim(img)) > 0
})

excluded_df <- img_df %>%
  mutate(is_valid = validity) %>%
  filter(!is_valid) %>%
  transmute(path, label, reason = "Unreadable or empty image")

img_df <- img_df %>% filter(validity)
message("After removing unreadable images: ", nrow(img_df))
print(table(img_df$label))

# Remove duplicate images by content hash
compute_hash <- function(p, size = 64) {
  img <- safe_load_image(p)
  if (is.null(img)) return(NA_character_)
  img_small <- imager::resize(img, size_x = size, size_y = size)

  if (imager::spectrum(img_small) == 3) img_small <- imager::grayscale(img_small)
  m <- as.numeric(img_small[,,1,1])
  digest::digest(m, algo = "xxhash64")
}

hashes <- map_chr(img_df$path, compute_hash)
dup_df <- img_df %>%
  mutate(hash = hashes) %>%
  filter(!is.na(hash)) %>%
  group_by(hash) %>%
  mutate(dup_rank = row_number()) %>%
  ungroup()

dups_removed <- dup_df %>% filter(dup_rank > 1) %>%
  transmute(path, label, reason = "Duplicate content (hash match)")

img_df <- dup_df %>% filter(dup_rank == 1) %>% select(path, label)

excluded_df <- bind_rows(excluded_df, dups_removed)

message("After removing duplicates: ", nrow(img_df))
print(table(img_df$label))

# Save excluded files log 
excluded_file <- file.path(output_dir, "preprocess_excluded_files.csv")
write_csv(excluded_df, excluded_file)
message("Saved excluded files log: ", excluded_file)

# Reproducible stratified train/test split
set.seed(SEED)
train_idx <- caret::createDataPartition(img_df$label, p = 1 - TEST_P, list = FALSE)
img_df <- img_df %>%
  mutate(split = if_else(row_number() %in% train_idx, "train", "test"))

message("Split counts:")
print(table(img_df$split, img_df$label))


# 2) IMAGE HELPERS

read_resize_cimg <- function(path, size = IMG_SIZE) {
  img <- safe_load_image(path)
  if (is.null(img)) stop("Could not load image: ", path)
  imager::resize(img, size_x = size, size_y = size)
}

# Ensure RGB for HSV (handles grayscale or 4-channel images)
ensure_rgb <- function(img_cimg) {
  s <- imager::spectrum(img_cimg)
  
  if (s == 1) {
    img_cimg <- imager::as.cimg(array(rep(as.vector(img_cimg), 3),
                                      dim = c(dim(img_cimg)[1], dim(img_cimg)[2], 1, 3)))
  }
  
  if (imager::spectrum(img_cimg) > 3) {
    img_cimg <- img_cimg[,,,1:3]
  }
  
  img_cimg
}

cimg_to_gray_matrix <- function(img_cimg) {
  if (imager::spectrum(img_cimg) == 3) img_cimg <- imager::grayscale(img_cimg)
  as.matrix(img_cimg[,,1,1])
}

# 3) FEATURE EXTRACTORS

# Uniform LBP 
build_uniform_lbp_lut <- function() {
  transitions8 <- function(bits) sum(bits != c(bits[-1], bits[1]))
  uniform_codes <- c()
  
  for (code in 0:255) {
    bits <- as.integer(intToBits(code))[1:8]
    if (transitions8(bits) <= 2) uniform_codes <- c(uniform_codes, code)
  }
  uniform_codes <- sort(unique(uniform_codes))  # typically 58
  lut <- integer(256)
  for (i in seq_along(uniform_codes)) lut[uniform_codes[i] + 1] <- i
  lut[lut == 0] <- length(uniform_codes) + 1   # non-uniform -> 59
  lut
}
ULUT <- build_uniform_lbp_lut()

lbp_8_1_codes <- function(m) {
  nr <- nrow(m); nc <- ncol(m)
  out <- matrix(0L, nr, nc)
  
  dx <- c(-1,-1,-1, 0, 1, 1, 1, 0)
  dy <- c(-1, 0, 1, 1, 1, 0,-1,-1)
  
  for (r in 2:(nr-1)) {
    for (c in 2:(nc-1)) {
      center <- m[r, c]
      code <- 0L
      for (k in 1:8) {
        bit <- as.integer(m[r + dx[k], c + dy[k]] >= center)
        code <- code + bit * bitwShiftL(1L, k-1L)
      }
      out[r, c] <- code
    }
  }
  out
}

uniform_lbp_hist59 <- function(lbp_codes, lut = ULUT) {
  bins <- lut[as.vector(lbp_codes) + 1]   # 1..59
  h <- tabulate(bins, nbins = 59)
  h / (sum(h) + 1e-8)
}

extract_spatial_ulbp_2x2 <- function(gray_mat) {
  size <- nrow(gray_mat)
  half <- size %/% 2
  
  TL <- gray_mat[1:half, 1:half]
  TR <- gray_mat[1:half, (half+1):size]
  BL <- gray_mat[(half+1):size, 1:half]
  BR <- gray_mat[(half+1):size, (half+1):size]
  
  c(
    uniform_lbp_hist59(lbp_8_1_codes(TL)),
    uniform_lbp_hist59(lbp_8_1_codes(TR)),
    uniform_lbp_hist59(lbp_8_1_codes(BL)),
    uniform_lbp_hist59(lbp_8_1_codes(BR))
  )
}

# HSV histogram
hsv_hist <- function(img_cimg, h_bins = H_BINS, s_bins = S_BINS, v_bins = V_BINS) {
  img_cimg <- ensure_rgb(img_cimg)
  
  hsv_img <- imager::RGBtoHSV(img_cimg)
  
  H <- as.vector(hsv_img[,,1,1])
  S <- as.vector(hsv_img[,,1,2])
  V <- as.vector(hsv_img[,,1,3])
  
  clamp01 <- function(x) {
    x <- x[is.finite(x)]
    x[x < 0] <- 0
    x[x > 1] <- 1
    x
  }
  
  H <- clamp01(H); S <- clamp01(S); V <- clamp01(V)
  
  h <- hist(H, breaks = seq(0, 1, length.out = h_bins + 1),
            plot = FALSE, include.lowest = TRUE, right = FALSE)$counts
  s <- hist(S, breaks = seq(0, 1, length.out = s_bins + 1),
            plot = FALSE, include.lowest = TRUE, right = FALSE)$counts
  v <- hist(V, breaks = seq(0, 1, length.out = v_bins + 1),
            plot = FALSE, include.lowest = TRUE, right = FALSE)$counts
  
  feat <- c(h, s, v)
  feat / (sum(feat) + 1e-8)
}

# HOG
hog_extract <- function(gray_mat, cells = HOG_CELLS, orientations = HOG_ORIENTATIONS) {
  h <- OpenImageR::HOG(image = gray_mat, cells = cells, orientations = orientations)
  as.numeric(h)
}

# Full feature extractor per image 
extract_features_one <- function(path) {
  img  <- read_resize_cimg(path, size = IMG_SIZE)
  gray <- cimg_to_gray_matrix(img)
  
  ulbp236 <- extract_spatial_ulbp_2x2(gray)
  hsv16   <- hsv_hist(img)
  hog_vec <- hog_extract(gray)
  
  c(ulbp236, hog_vec, hsv16)
}

safe_extract_features_one <- purrr::possibly(extract_features_one, otherwise = NULL)


# 4) EXTRACTION + SANITY CHECKS

features <- map(img_df$path, safe_extract_features_one)

ok <- map_lgl(features, ~ !is.null(.x))
if (any(!ok)) {
  excluded_df2 <- img_df[!ok, ] %>%
    transmute(path, label, reason = "Feature extraction failed")
  excluded_df <- bind_rows(excluded_df, excluded_df2)
  write_csv(excluded_df, excluded_file)
}

img_df_ok <- img_df[ok, ]
feature_matrix <- do.call(rbind, features[ok])
features_df <- as_tibble(feature_matrix)

# Build column names after we know HOG length
hog_len <- ncol(feature_matrix) - 236 - 16
colnames(features_df) <- c(
  paste0("ulbp_", 1:236),
  paste0("hog_", 1:hog_len),
  paste0("hsv_", 1:16)
)

final_df <- bind_cols(img_df_ok, features_df)

hsv_sum <- rowSums(select(final_df, starts_with("hsv_")))

ulbp_sum <- rowSums(select(final_df, starts_with("ulbp_")))

stopifnot(all(is.finite(hsv_sum)))
stopifnot(all(is.finite(ulbp_sum)))
stopifnot(all(abs(hsv_sum - 1) < 1e-4))
stopifnot(all(abs(ulbp_sum - 4) < 1e-3))

message("Sanity checks passed: HSV sums to 1, uLBP sums to 4.")


# 5) SAVE OUTPUTS

out_all <- file.path(output_dir, "features_spatial_ulbp_hog_hsv.csv")
write_csv(final_df, out_all)
message("Saved full features: ", out_all)
print(dim(final_df))

# Save train/test as separate files (reproducible)
train_df <- final_df %>% filter(split == "train")
test_df  <- final_df %>% filter(split == "test")

out_train <- file.path(output_dir, "features_train.csv")
out_test  <- file.path(output_dir, "features_test.csv")

write_csv(train_df, out_train)
write_csv(test_df, out_test)

message("Saved train features: ", out_train)
message("Saved test features:  ", out_test)
message("Done.")


# 6) Model training

# Paths and settings
train_file <- here::here("outputs", "features_train.csv")
test_file  <- here::here("outputs", "features_test.csv")
out_dir    <- here::here("outputs")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

set.seed(123)
positive_class <- "no_helmet" 


# Helper functions
metrics_from_cm <- function(cm, positive = "no_helmet") {

  labels <- rownames(cm)
  if (!(positive %in% labels) || !(positive %in% colnames(cm))) {
    stop("Positive class not found in confusion matrix.")
  }
  
  tp <- cm[positive, positive]
  fp <- sum(cm[positive, ]) - tp
  fn <- sum(cm[, positive]) - tp
  tn <- sum(cm) - tp - fp - fn
  
  precision <- tp / (tp + fp + 1e-9)
  recall    <- tp / (tp + fn + 1e-9)
  f1        <- 2 * precision * recall / (precision + recall + 1e-9)
  accuracy  <- (tp + tn) / sum(cm)
  specificity <- tn / (tn + fp + 1e-9)
  
  tibble(
    accuracy = accuracy,
    precision_no_helmet = precision,
    recall_no_helmet = recall,
    f1_no_helmet = f1,
    specificity_helmet = specificity,
    tp = as.numeric(tp),
    fp = as.numeric(fp),
    fn = as.numeric(fn),
    tn = as.numeric(tn)
  )
}

save_results <- function(model_name, pred_class, pred_prob = NULL, truth, test_meta) {
  cm <- table(Predicted = pred_class, Actual = truth)
  metrics <- metrics_from_cm(cm, positive = positive_class) %>%
    mutate(model = model_name, .before = 1)
  
  write_csv(metrics, file.path(out_dir, paste0("metrics_", model_name, ".csv")))
  
  cm_df <- as.data.frame(cm)
  write_csv(cm_df, file.path(out_dir, paste0("confusion_", model_name, ".csv")))
  
  pred_df <- test_meta %>%
    mutate(actual = truth,
           predicted = pred_class)
  
  if (!is.null(pred_prob)) {
    pred_df <- pred_df %>% mutate(prob_no_helmet = pred_prob)
  }
  
  write_csv(pred_df, file.path(out_dir, paste0("predictions_", model_name, ".csv")))
  
  list(metrics = metrics, confusion = cm)
}

# Load data
train_df <- read_csv(train_file, show_col_types = FALSE)
test_df  <- read_csv(test_file,  show_col_types = FALSE)

required_cols <- c("label")
missing_train <- setdiff(required_cols, names(train_df))
missing_test  <- setdiff(required_cols, names(test_df))
if (length(missing_train) > 0) stop("Missing columns in training data: ", paste(missing_train, collapse = ", "))
if (length(missing_test) > 0)  stop("Missing columns in test data: ", paste(missing_test, collapse = ", "))

meta_cols <- intersect(c("path", "label", "split"), names(test_df))
test_meta <- test_df %>% select(any_of(meta_cols))

train_df <- train_df %>% mutate(label = factor(label, levels = c("helmet", "no_helmet")))
test_df  <- test_df  %>% mutate(label = factor(label, levels = c("helmet", "no_helmet")))

non_feature_cols <- c("path", "label", "split")
feature_cols <- names(train_df)[!(names(train_df) %in% non_feature_cols)]

train_df[feature_cols] <- lapply(train_df[feature_cols], as.numeric)
test_df[feature_cols]  <- lapply(test_df[feature_cols], as.numeric)

for (col in feature_cols) {
  train_df[[col]][!is.finite(train_df[[col]])] <- NA_real_
  test_df[[col]][!is.finite(test_df[[col]])] <- NA_real_
}

train_medians <- sapply(train_df[feature_cols], function(x) median(x, na.rm = TRUE))
for (col in feature_cols) {
  train_df[[col]][is.na(train_df[[col]])] <- train_medians[[col]]
  test_df[[col]][is.na(test_df[[col]])]   <- train_medians[[col]]
}

# Matrices for modelling
x_train <- train_df %>% select(all_of(feature_cols))
x_test  <- test_df  %>% select(all_of(feature_cols))
y_train <- train_df$label
y_test  <- test_df$label

# Scale features using TRAIN ONLY
preproc <- caret::preProcess(x_train, method = c("center", "scale"))
x_train_scaled <- predict(preproc, x_train)
x_test_scaled  <- predict(preproc, x_test)

# Class imbalance handling

train_counts <- table(y_train)

class_weights <- c(
  helmet = as.numeric(sum(train_counts) / (2 * train_counts["helmet"])),
  no_helmet = as.numeric(sum(train_counts) / (2 * train_counts["no_helmet"]))
)

# 7) Model A: Linear SVM (primary model)
linear_cost_grid <- c(0.01, 0.1, 1, 5, 10)

linear_tuned <- tune.svm(
  x = x_train_scaled,
  y = y_train,
  kernel = "linear",
  cost = linear_cost_grid,
  class.weights = class_weights,
  tunecontrol = tune.control(cross = 5)
)

svm_linear <- linear_tuned$best.model

linear_pred <- predict(svm_linear, newdata = x_test_scaled, decision.values = TRUE)
linear_decision <- attributes(linear_pred)$decision.values
linear_prob_like <- as.numeric(linear_decision)  # distance from margin, useful for ranking

linear_results <- save_results(
  model_name = "svm_linear",
  pred_class = linear_pred,
  pred_prob = linear_prob_like,
  truth = y_test,
  test_meta = test_meta
)
cv_linear <- linear_tuned$performances
cv_linear$mean_cv_accuracy <- 1 - cv_linear$error

print(cv_linear)

readr::write_csv(
  cv_linear,
  file.path(out_dir, "svm_linear_cv_results.csv")
)

# 8) Model B: RBF SVM (secondary model)
rbf_cost_grid  <- c(0.1, 1, 5, 10)
rbf_gamma_grid <- c(0.001, 0.01, 0.05, 0.1)

rbf_tuned <- tune.svm(
  x = x_train_scaled,
  y = y_train,
  kernel = "radial",
  cost = rbf_cost_grid,
  gamma = rbf_gamma_grid,
  class.weights = class_weights,
  tunecontrol = tune.control(cross = 5)
)

svm_rbf <- rbf_tuned$best.model

rbf_pred <- predict(svm_rbf, newdata = x_test_scaled, decision.values = TRUE)
rbf_decision <- attributes(rbf_pred)$decision.values
rbf_prob_like <- as.numeric(rbf_decision)

rbf_results <- save_results(
  model_name = "svm_rbf",
  pred_class = rbf_pred,
  pred_prob = rbf_prob_like,
  truth = y_test,
  test_meta = test_meta
)
cv_rbf <- rbf_tuned$performances
cv_rbf$mean_cv_accuracy <- 1 - cv_rbf$error
print(cv_rbf)
readr::write_csv(cv_rbf, file.path(out_dir, "svm_rbf_cv_results.csv"))

# 9) Console output
cat("\n================ SVM RESULTS ================\n")
cat("Train class counts:\n")
print(train_counts)
cat("\nTest class counts:\n")
print(table(y_test))

cat("\nBest Linear SVM tuning:\n")
print(linear_tuned$best.parameters)
cat("\nLinear SVM confusion matrix:\n")
print(linear_results$confusion)
cat("\nLinear SVM metrics:\n")
print(linear_results$metrics)


cat("\nBest RBF SVM tuning:\n")
print(rbf_tuned$best.parameters)
cat("\nRBF SVM confusion matrix:\n")
print(rbf_results$confusion)
cat("\nRBF SVM metrics:\n")
print(rbf_results$metrics)

# 10) Save concise text summary for report writing
summary_lines <- c(
  "SVM training summary",
  "====================",
  paste("Training rows:", nrow(train_df)),
  paste("Test rows:", nrow(test_df)),
  paste("Feature count:", length(feature_cols)),
  paste("Positive class:", positive_class),
  "",
  "Linear SVM best parameters:",
  paste(capture.output(print(linear_tuned$best.parameters)), collapse = " "),
  "",
  "Linear SVM metrics:",
  paste(capture.output(print(linear_results$metrics)), collapse = " "),
  "",
  "RBF SVM best parameters:",
  paste(capture.output(print(rbf_tuned$best.parameters)), collapse = " "),
  "",
  "RBF SVM metrics:",
  paste(capture.output(print(rbf_results$metrics)), collapse = " ")
)
writeLines(summary_lines, con = file.path(out_dir, "svm_summary.txt"))

cat("\nSaved outputs to:", out_dir, "\n") 
cat("Done.\n")
