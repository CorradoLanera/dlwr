library(reticulate)
library(tensorflow)
library(keras)

library(glue)
library(fs)
library(zip)



# Globals ---------------------------------------------------------

first_run <- FALSE



# Functions -------------------------------------------------------

make_subset <- function(subset_name, start_index, end_index) {
  for (category in c("dog", "cat")) {
    file_name <- glue("{category}.{start_index:end_index}.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(
      original_dir / file_name,
      new_base_dir / subset_name / category /  file_name
    )
  }
}



# Dataset ---------------------------------------------------------

original_dir <- path("dogs-vs-cats/train")
new_base_dir <- path("dogs-vs-cats_small")

if (first_run) {
  # accept rules at http://www.kaggle.com/c/dogs-vs-cats/rules
  system("kaggle competitions download -c dogs-vs-cats")
  unzip("dogs-vs-cats.zip", exdir = "dogs-vs-cats", files = "train.zip")
  unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")

  make_subset("train", start_index = 1, end_index = 1000)
  make_subset("validation", start_index = 1001, end_index = 1500)
  make_subset("test", start_index = 1501, end_index = 2500)
}

train_dataset <- image_dataset_from_directory(
  new_base_dir / "train",
  image_size = c(180, 180),
  batch_size = 32
)

validation_dataset <- image_dataset_from_directory(
  new_base_dir / "validation",
  image_size = c(180, 180),
  batch_size = 32
)

test_dataset <- image_dataset_from_directory(
  new_base_dir / "test",
  image_size = c(180, 180),
  batch_size = 32
)



# Model -----------------------------------------------------------

inputs <- layer_input(shape = c(180, 180, 3))

outputs <- inputs |>
  layer_rescaling(1/255) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_flatten() |>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)

model |>
  compile(
    loss = "binary_crossentropy",
    optimizer = "rmsprop",
    metrics = "accuracy"
  )



# Callbacks -------------------------------------------------------

callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_skratch_with.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)



# Fit -------------------------------------------------------------

history <- model |>
  fit(
    train_dataset,
    epochs = 100,
    validation_data = validation_dataset,
    callbacks = callbacks
  )



# Evaluate --------------------------------------------------------

test_model <- load_model_tf("convnet_from_skratch.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result[["accuracy"]]))



# Data augmentation -----------------------------------------------
data_augmentation <- keras_model_sequential() |>
  layer_random_flip("horizontal") |>
  layer_random_rotation(0.1) |>
  layer_random_zoom(0.2)


outputs_aumented <- inputs |>
  data_augmentation() |>
  layer_rescaling(1/255) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_flatten() |>
  layer_dropout(0.5) |>
  layer_dense(1, activation = "sigmoid")

model_augmented <- keras_model(inputs, outputs_aumented)

model_augmented |>
  compile(
    loss = "binary_crossentropy",
    optimizer = "rmsprop",
    metrics = "accuracy"
  )

callbacks_augmented <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_skratch_with_augmentation.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)


history <- model_augmented |>
  fit(
    train_dataset,
    epochs = 100,
    validation_data = validation_dataset,
    callbacks = callbacks_augmented
  )



# Evaluate --------------------------------------------------------

test_model_augmented <- load_model_tf("convnet_from_skratch_with_augmentation.keras")
result <- evaluate(test_model_augmented, test_dataset)
cat(sprintf("Augmented test accuracy: %.3f\n", result[["accuracy"]]))
